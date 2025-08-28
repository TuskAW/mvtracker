"""
SelfCap dataset (https://zju3dv.github.io/longvolcap/)

Download the dataset (but first fill in the form at https://forms.gle/MzJqZjBfyZ53fRMZ7):
```bash
mkdir -p datasets/selfcap
cd datasets/selfcap

gdown --fuzzy https://drive.google.com/file/d/1iTr6sTVQoCtTK4FbA3lRxMrh7sC0MhzP/view?usp=share_link  # LICENSE
gdown --fuzzy https://drive.google.com/file/d/1cg54hE_IBsnVXuMCj44JCQEGnqU1Hr5b/view?usp=share_link  # yoga-calib.tar.gz
gdown --fuzzy https://drive.google.com/file/d/1l84Pna4eO9m_bql2mR8nm6VnLO80e717/view?usp=share_link  # hair-calib.tar.gz
gdown --fuzzy https://drive.google.com/file/d/1Desj7th500-vsyRYzRq8Xb6TtUgDPU4u/view?usp=share_link  # README.md
gdown --fuzzy https://drive.google.com/file/d/1Ex3OtLmz6kBbgB84MImlDLJpVE6vI3ks/view?usp=share_link  # bike-release.tar.gz
gdown --fuzzy https://drive.google.com/file/d/1muPLxdCm4il_X6TRVLaxx-6sYO6XYIwH/view?usp=share_link  # yoga-release.tar.gz
gdown --fuzzy https://drive.google.com/file/d/12mRUCpaTk1XearBq2hUIf5ZbHZw4AQAw/view?usp=share_link  # dance-release.tar.gz
gdown --fuzzy https://drive.google.com/file/d/1AEiQBC9CIthR97qZeZzkH2nlXXpogfxH/view?usp=share_link  # hair-release.tar.gz
gdown --fuzzy https://drive.google.com/file/d/1NFrHh-SxUER4jWBV0irnCcDhEmkg3WUg/view?usp=share_link  # corgi-release.tar.gz
gdown --fuzzy https://drive.google.com/file/d/1b9Hf3YY_usPrtddgpMe569dSqh0bEGLo/view?usp=share_link  # bar-release.tar.gz

tar xvf bar-release.tar.gz
tar xvf bike-release.tar.gz
tar xvf corgi-release.tar.gz
tar xvf dance-release.tar.gz
tar xvf hair-calib.tar.gz
tar xvf hair-release.tar.gz
tar xvf yoga-calib.tar.gz
tar xvf yoga-release.tar.gz

rm *.tar.gz

cd -
```
Running the script: `PYTHONPATH=/local/home/frrajic/xode/duster:$PYTHONPATH python -m scripts.selfcap_preprocessing`
Note that you need to set up dust3r first, see docstring of `scripts/estimate_depth_with_duster.py`.
"""

import concurrent.futures
import json
import os
import pickle
from typing import Optional

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from scripts.egoexo4d_preprocessing import main_estimate_duster_depth


def main_preprocess_selfcap(
        dataset_root: str,
        scene_name: str,
        outputs_dir: str,
        num_cameras: Optional[int] = None,
        sample_cameras_sequentially: Optional[bool] = False,
        start_frame: Optional[int] = None,
        max_frames: Optional[int] = None,
        frames_downsampling_factor: Optional[int] = None,
        downscaled_longerside: Optional[int] = None,
        save_rerun_viz: bool = True,
        stream_rerun_viz: bool = False,
        skip_if_output_exists: bool = True,
):
    # Skip if output exists
    save_pkl_path = os.path.join(outputs_dir, f"{scene_name}.pkl")
    if skip_if_output_exists and os.path.exists(save_pkl_path):
        print(f"Skipping {save_pkl_path} since it already exists")
        print()
        return save_pkl_path
    else:
        print(f"Processing {scene_name}...")

    # --- Load calibration ---
    calib_dir = os.path.join(dataset_root, f"{scene_name}-calib", "optimized")
    intri_path = os.path.join(calib_dir, "intri.yml")
    extri_path = os.path.join(calib_dir, "extri.yml")
    sync_path = os.path.join(calib_dir, "sync.json")

    assert all(os.path.exists(p) for p in [intri_path, extri_path, sync_path])

    intri_fs = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    extri_fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    with open(sync_path) as f:
        sync_data = json.load(f)

    # --- Load videos ---
    video_dir = os.path.join(dataset_root, f"{scene_name}-release", "videos")
    cam_names = sorted([f.replace(".mp4", "") for f in os.listdir(video_dir) if f.endswith(".mp4")])

    if num_cameras is not None and num_cameras < len(cam_names):
        if sample_cameras_sequentially:
            cam_names = cam_names[:num_cameras]
        else:
            step = len(cam_names) / num_cameras
            cam_names = [cam_names[int(i * step)] for i in range(num_cameras)]

    rgbs, intrs, extrs = {}, {}, {}

    def load_cam_video(cam):
        vid_path = os.path.join(video_dir, f"{cam}.mp4")
        cap = cv2.VideoCapture(vid_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        offset = int(round(sync_data[cam] * fps))

        frames = []
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            idx = i - offset
            i += 1
            if idx < 0:
                continue
            if start_frame is not None and idx < start_frame:
                continue
            if frames_downsampling_factor and ((idx - start_frame) % frames_downsampling_factor != 0):
                continue
            if max_frames and len(frames) >= max_frames:
                break
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            frames.append(img)
        cap.release()

        if not frames:
            return None, None, None

        rgb = np.stack(frames)
        intr = intri_fs.getNode(f"K_{cam}").mat().astype(np.float32)
        R = extri_fs.getNode(f"Rot_{cam}").mat().astype(np.float32)
        T = extri_fs.getNode(f"T_{cam}").mat().astype(np.float32).reshape(3)
        extr = np.concatenate([R, T[:, None]], axis=1)

        return cam, rgb, intr, extr

    # Run parallel loading
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_cam_video, cam) for cam in cam_names]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            cam, rgb, intr, extr = future.result()
            if cam is None:
                print("Warning: camera skipped due to no usable frames.")
                continue
            rgbs[cam] = rgb
            intrs[cam] = intr
            extrs[cam] = extr

    intri_fs.release()
    extri_fs.release()

    # Apply a global -90° rotation around X axis to the scene
    rot_x = R.from_euler('x', -90, degrees=True).as_matrix()
    rot_y = R.from_euler('y', 0, degrees=True).as_matrix()
    rot_z = R.from_euler('z', 0, degrees=True).as_matrix()
    rot = rot_z @ rot_y @ rot_x
    T_rot = np.eye(4)
    T_rot[:3, :3] = rot
    for cam in extrs:
        extrs_square = np.eye(4, dtype=extrs[cam].dtype)
        extrs_square[:3, :] = extrs[cam]
        extrs_trans_square = np.einsum('ki,ij->kj', extrs_square, T_rot.T)
        extrs_trans = extrs_trans_square[..., :3, :]
        assert np.allclose(extrs_trans_square[..., 3, 3], np.ones_like(extrs_trans_square[..., 3, 3]))
        extrs[cam] = extrs_trans

    print(f"Loaded SelfCap scene '{scene_name}' with {len(cam_names)} cams and {rgbs[cam_names[0]].shape[0]} frames.")

    # Check shapes
    n_frames, _, h, w = rgbs[cam_names[0]].shape
    for cam_name in cam_names:
        assert rgbs[cam_name].shape == (n_frames, 3, h, w)
        assert intrs[cam_name].shape == (3, 3)
        assert extrs[cam_name].shape == (3, 4)

    # Save downsized version
    if downscaled_longerside is not None:
        print(f"Downscaling to longer side {downscaled_longerside}")
        for cam_name in tqdm(cam_names, desc="Downscaling"):
            _, _, h, w = rgbs[cam_name].shape
            scale = downscaled_longerside / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            resized = []
            for img in rgbs[cam_name]:
                img = img.transpose(1, 2, 0)  # CHW -> HWC
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized.append(img.transpose(2, 0, 1))  # HWC -> CHW
            rgbs[cam_name] = np.stack(resized)

            # scale intrinsics
            intrs[cam_name][:2] *= scale

    # Save processed output to a pickle file
    os.makedirs(outputs_dir, exist_ok=True)
    with open(save_pkl_path, "wb") as f:
        pickle.dump(
            dict(
                rgbs=rgbs,
                intrs=intrs,
                extrs=extrs,
                ego_cam_name=None,
            ),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {save_pkl_path}")

    # Visualize the data sample using rerun
    rerun_modes = []
    if stream_rerun_viz:
        rerun_modes += ["stream"]
    if save_rerun_viz:
        rerun_modes += ["save"]
    for rerun_mode in rerun_modes:
        rr.init(f"3dpt", recording_id="v0.16")
        if rerun_mode == "stream":
            rr.connect_tcp()

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.set_time_seconds("frame", 0)
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

        fps = 30
        for frame_idx in range(min(n_frames, 30)):
            rr.set_time_seconds("frame", frame_idx / fps)

            for cam_name in cam_names:
                extr = extrs[cam_name]
                intr = intrs[cam_name]
                img = rgbs[cam_name][frame_idx].transpose(1, 2, 0).astype(np.uint8)

                # Camera pose logging
                E = extr if extr.shape == (3, 4) else extr[0]
                T = np.eye(4)
                T[:3, :] = E
                T_world_cam = np.linalg.inv(T)
                rr.log(f"{cam_name}/image", rr.Transform3D(
                    translation=T_world_cam[:3, 3],
                    mat3x3=T_world_cam[:3, :3],
                ))

                # Intrinsics and image
                rr.log(f"{cam_name}/image", rr.Pinhole(
                    image_from_camera=intr,
                    width=img.shape[1],
                    height=img.shape[0]
                ))
                rr.log(f"{cam_name}/image", rr.Image(img))

        if rerun_mode == "save":
            save_rrd_path = os.path.join(outputs_dir, f"rerun__{scene_name}.rrd")
            rr.save(save_rrd_path)
            print(f"Saved rerun viz to {os.path.abspath(save_rrd_path)}")

    return save_pkl_path


if __name__ == '__main__':
    dataset_root = "datasets/selfcap/"
    outputs_dir = "datasets/selfcap-processed/"

    for scene_name in ["yoga", "hair"]:
        for num_cameras, sequential_cams, start_frame, max_frames, frames_downsampling_factor, downscaled_longerside in [
            (8, False, 90, 256, 10, 512),
            (8, True, 90, 256, 10, 512),
            (8, False, 90, 2560, 10, 512),

            (4, False, 90, 256, 10, 512),
            (4, True, 90, 256, 10, 512),

            (16, False, 90, 256, 10, 512),
            (16, True, 90, 256, 10, 512),
            (16, True, 90, 2560, 10, 512),

            (8, False, 90, 256, 1, 512),
            (8, False, 90, 2560, 1, 512),
            (8, False, 90, 256, 5, 512),
            (8, False, 90, 256, 20, 512),
            (8, False, 90, 256, 30, 512),
        ]:
            # Extract rgbs, intrs, extrs from SelfCap
            outputs_subdir = os.path.join(
                outputs_dir, f"numcams-{num_cameras}-seq-{sequential_cams}_"
                             f"startframe-{start_frame}_"
                             f"maxframes-{max_frames}_"
                             f"downsample-{frames_downsampling_factor}_"
                             f"downscale-{downscaled_longerside}"
            )
            scene_pkl = main_preprocess_selfcap(
                dataset_root=dataset_root,
                scene_name=scene_name,
                outputs_dir=outputs_subdir,
                num_cameras=num_cameras,
                sample_cameras_sequentially=sequential_cams,
                start_frame=start_frame,
                max_frames=max_frames,
                frames_downsampling_factor=frames_downsampling_factor,
                downscaled_longerside=downscaled_longerside,
            )

            # Run Dust3r to estimate depths from rgbs, fix the known intrs and extrs during multi-view stereo optim
            depth_subdir = os.path.join(outputs_subdir, f"duster_depths__{scene_name}")
            main_estimate_duster_depth(
                pkl_scene_file=scene_pkl,
                depths_output_dir=depth_subdir,
            )

            # Run VGGT to estimate depths from rgbs, align with the known extrs afterward
            ...
