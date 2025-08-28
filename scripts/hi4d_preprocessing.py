"""
First download the dataset. You'll have to fill in an online ETH form
and then wait for a few days to get a temporary access code over email.
I used the following sequence of commands to download and unpack the data
into the expected structure. You can probably replace the `dt=...` with
your access token that you can probably find in the access URL (or otherwise
in the page source of the download page that will be linked). Note that
you don't need to download all the data if you don't need it, e.g., maybe
you just want to download a small sample. Note also that in the commands below,
I didn't delete the `*.tar.gz` files, but you can do so if you'd like.
```bash
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/LICENSE.txt' -O LICENSE.txt
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/README.md' -O README.md
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair00_1.tar.gz' -O pair00_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair00_2.tar.gz' -O pair00_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair01.tar.gz' -O pair01.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair02_1.tar.gz' -O pair02_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair02_2.tar.gz' -O pair02_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair09.tar.gz' -O pair09.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair10.tar.gz' -O pair10.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair12.tar.gz' -O pair12.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair13_1.tar.gz' -O pair13_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair13_2.tar.gz' -O pair13_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair14.tar.gz' -O pair14.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair15_1.tar.gz' -O pair15_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair15_2.tar.gz' -O pair15_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair16.tar.gz' -O pair16.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair17_1.tar.gz' -O pair17_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair17_2.tar.gz' -O pair17_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair18_1.tar.gz' -O pair18_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair18_2.tar.gz' -O pair18_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair19_1.tar.gz' -O pair19_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair19_2.tar.gz' -O pair19_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair21_1.tar.gz' -O pair21_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair21_2.tar.gz' -O pair21_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair22.tar.gz' -O pair22.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair23_1.tar.gz' -O pair23_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair23_2.tar.gz' -O pair23_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair27_1.tar.gz' -O pair27_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair27_2.tar.gz' -O pair27_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair28.tar.gz' -O pair28.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair32_1.tar.gz' -O pair32_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair32_2.tar.gz' -O pair32_2.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair37_1.tar.gz' -O pair37_1.tar.gz
wget 'https://hi4d.ait.ethz.ch/download.php?dt=def502001190eca4e725f10acbfbd3520f0caca29004163d940aa67e31c024acac2f55ce060924e95b528e99e47e167d6d3e8dd34449e7c89fc60b1139e6ee28f45ed216e5f452230156127a2a1919ef0b796c8cc016630353296abd0c4294db83582d7a99a132d033e95928e4a1&file=/pair37_2.tar.gz' -O pair37_2.tar.gz

mkdir -p pair00 pair01 pair02 pair09 pair10 pair12 pair13 pair14 pair15 pair16 pair17 pair18 pair19 pair21 pair22 pair23 pair27 pair28 pair32 pair37

tar -xvzf pair00_1.tar.gz -C pair00
tar -xvzf pair00_2.tar.gz -C pair00
tar -xvzf pair01.tar.gz pair01
tar -xvzf pair02_1.tar.gz -C pair02
tar -xvzf pair02_2.tar.gz -C pair02
tar -xvzf pair09.tar.gz -C pair09
tar -xvzf pair10.tar.gz -C pair10
tar -xvzf pair12.tar.gz -C pair12
tar -xvzf pair13_1.tar.gz -C pair13
tar -xvzf pair13_2.tar.gz -C pair13
tar -xvzf pair14.tar.gz -C pair14
tar -xvzf pair15_1.tar.gz -C pair15
tar -xvzf pair15_2.tar.gz -C pair15
tar -xvzf pair16.tar.gz -C pair16
tar -xvzf pair17_1.tar.gz -C pair17
tar -xvzf pair17_2.tar.gz -C pair17
tar -xvzf pair18_1.tar.gz -C pair18
tar -xvzf pair18_2.tar.gz -C pair18
tar -xvzf pair19_1.tar.gz -C pair19
tar -xvzf pair19_2.tar.gz -C pair19
tar -xvzf pair21_1.tar.gz -C pair21
tar -xvzf pair21_2.tar.gz -C pair21
tar -xvzf pair22.tar.gz -C pair22
tar -xvzf pair23_1.tar.gz -C pair23
tar -xvzf pair23_2.tar.gz -C pair23
tar -xvzf pair27_1.tar.gz -C pair27
tar -xvzf pair27_2.tar.gz -C pair27
tar -xvzf pair28.tar.gz -C pair28
tar -xvzf pair32_1.tar.gz -C pair32
tar -xvzf pair32_2.tar.gz -C pair32
tar -xvzf pair37_1.tar.gz -C pair37
tar -xvzf pair37_2.tar.gz -C pair37

# Some cleanup because the tars were not consistently structured
mv pair00/pair00/* pair00/
mv pair01/pair01/* pair01/
mv pair02/pair02/* pair02/
mv pair09/pair09/* pair09/
mv pair10/pair10/* pair10/
mv pair12/pair12/* pair12/
mv pair13/pair13/* pair13/
mv pair14/pair14/* pair14/
mv pair15/pair15/* pair15/
mv pair16/pair16/* pair16/
mv pair17/pair17/* pair17/
mv pair18/pair18/* pair18/
mv pair19/pair19/* pair19/
mv pair21/pair21/* pair21/
mv pair22/pair22/* pair22/
mv pair23/pair23/* pair23/
mv pair27/pair27/* pair27/
mv pair28/pair28/* pair28/
mv pair32/pair32/* pair32/
mv pair37/pair37/* pair37/
rm -rf pair*/pair*/
```

With the data downloaded, you can run the script: `python -m scripts.hi4d_preprocessing`.
"""
from mvtracker.datasets.utils import transform_scene


def load_pickle(p):
    with open(p, "rb") as f:
        return pickle.load(f)


import glob
import os
import pickle
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import rerun as rr
import torch
import tqdm
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation


def save_pickle(p, data):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(data, f)


def load_image(path):
    return np.array(Image.open(path))


def _safe_load_rgb_cameras(npz_path: str) -> Dict[str, np.ndarray]:
    """
    Hi4D has a typo in docs ('intirnsics'). Support both.
    Returns dict with keys: ids [N], intrinsics [N,3,3], extrinsics [N,3,4], dist_coeffs [N,5]
    """
    data = dict(np.load(npz_path))
    ids = data.get("ids")
    intr = data.get("intrinsics", data.get("intirnsics"))
    extr = data.get("extrinsics")
    dist = data.get("dist_coeffs")
    assert ids is not None and intr is not None and extr is not None, \
        f"Missing keys in {npz_path}. Found keys: {list(data.keys())}"
    return {"ids": ids, "intrinsics": intr, "extrinsics": extr, "dist_coeffs": dist}


def _find_all_frames_for_action(images_root: str, cam_ids: List[int]) -> List[int]:
    """
    Robustly infer the list of frame indices by intersecting the available frames across cams.
    Hi4D names images as 000XXX.jpg (zero-padded 6).
    """
    per_cam_sets = []
    for cid in cam_ids:
        cam_dir = os.path.join(images_root, f"{cid}")
        jpgs = sorted(glob.glob(os.path.join(cam_dir, "*.jpg")))
        frames = set(int(os.path.splitext(os.path.basename(p))[0]) for p in jpgs)
        per_cam_sets.append(frames)
    if not per_cam_sets:
        return []
    common = set.intersection(*per_cam_sets) if len(per_cam_sets) > 1 else per_cam_sets[0]
    return sorted(list(common))


def _mesh_path_for_frame(frames_dir: str, frame_idx: int) -> str:
    """
    Hi4D meshes are 'mesh-f00XXX.obj' (5 digits). We'll format with 5 digits.
    """
    return os.path.join(frames_dir, f"mesh-f{frame_idx:05d}.obj")


def extract_hi4d_action_to_pkl(
        dataset_root: str,
        pair: str,
        action: str,
        save_pkl_path: str,
        downscaled_longerside: Optional[int] = None,
        save_rerun_viz: bool = True,
        stream_rerun_viz: bool = False,
        skip_if_output_exists: bool = False,
):
    """
    Build a single .pkl for a (pair, action):
      - rgbs:  dict[cam_id_str] -> [T,3,H,W] uint8
      - intrs: dict[cam_id_str] -> [3,3] float32  (scaled if resized)
      - extrs: dict[cam_id_str] -> [3,4] float32
      - depths:dict[cam_id_str] -> [T,H,W] float32  (mesh-rendered)
      - ego_cam_name: None
    """
    if skip_if_output_exists and os.path.exists(save_pkl_path):
        print(f"Skipping {save_pkl_path} (exists).")
        return save_pkl_path
    print(f"Processing {pair}/{action} -> {save_pkl_path}")

    root = os.path.join(dataset_root, pair, action)
    frames_dir = os.path.join(root, "frames")
    images_dir = os.path.join(root, "images")
    cameras_npz = os.path.join(root, "cameras", "rgb_cameras.npz")
    meta_npz = os.path.join(root, "meta.npz")

    cams = _safe_load_rgb_cameras(cameras_npz)
    cam_ids: List[int] = list(map(int, cams["ids"]))  # e.g., [4,16,28,40,52,64,76,88]
    intr_all = cams["intrinsics"].astype(np.float32)  # [N,3,3]
    extr_all = cams["extrinsics"].astype(np.float32)  # [N,3,4]

    meta = dict(np.load(meta_npz))
    frame_ids = _find_all_frames_for_action(images_dir, cam_ids)
    assert len(frame_ids) > 0, f"No common frames found across cameras at {images_dir}"
    assert frame_ids[0] == meta["start"].item()
    assert frame_ids[-1] == meta["end"].item()
    assert len(frame_ids) == (meta["end"].item() - meta["start"].item() + 1)

    # Build containers
    rgbs: Dict[str, List[np.ndarray]] = {str(cid): [] for cid in cam_ids}
    depths: Dict[str, List[np.ndarray]] = {str(cid): [] for cid in cam_ids}
    intrs: Dict[str, np.ndarray] = {}
    extrs: Dict[str, np.ndarray] = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pre-load a single mesh per frame and rasterize to each camera
    # (This is typically faster than reloading the mesh V times.)
    raster_settings_cache: Dict[Tuple[int, int], RasterizationSettings] = {}

    for frame in tqdm.tqdm(frame_ids, desc=f"Frames {pair}/{action}"):
        mesh_path = _mesh_path_for_frame(frames_dir, frame)
        if not os.path.isfile(mesh_path):
            # Some sequences may use different padding; try 6 digits as fallback.
            alt = os.path.join(frames_dir, f"mesh-f{frame:06d}.obj")
            if os.path.isfile(alt):
                mesh_path = alt
            else:
                # Skip missing mesh frame
                continue

        # Load mesh (geometry only is enough for depth)
        meshes: Meshes = load_objs_as_meshes([mesh_path], device=device)

        # For each camera, render depth & collect RGB
        for i, cid in enumerate(cam_ids):
            cam_name = str(cid)
            img_path = os.path.join(images_dir, cam_name, f"{frame:06d}.jpg")
            if not os.path.isfile(img_path):
                # Skip if that particular view is missing the image for this frame
                continue

            image = load_image(img_path)
            h0, w0 = image.shape[:2]

            # Copy camera params
            K = intr_all[i].copy()  # [3,3]
            E = extr_all[i].copy()  # [3,4]  world->cam (Hi4D)

            # Optional downscale (longer side) + scale intrinsics
            if downscaled_longerside is not None:
                scale = downscaled_longerside / float(max(h0, w0))
                nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
                if (nh, nw) != (h0, w0):
                    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
                    K[:2] *= scale
                h, w = nh, nw
            else:
                h, w = h0, w0

            # Stash static intr/extr once (raw, no global transform)
            if cam_name not in intrs:
                intrs[cam_name] = K.astype(np.float32)
                extrs[cam_name] = E.astype(np.float32)

            rgbs[cam_name].append(image)

            # Build PyTorch3D camera from raw E
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]

            R = E[:3, :3]
            t = E[:3, 3]

            # 4D-DRESS convention: transpose + flip X/Y
            R = R.T
            R = R @ np.diag(np.array([-1.0, -1.0, 1.0], dtype=np.float32))
            t = t @ np.diag(np.array([-1.0, -1.0, 1.0], dtype=np.float32))

            cameras_p3d = PerspectiveCameras(
                focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
                principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
                R=torch.tensor(R, dtype=torch.float32, device=device).unsqueeze(0),
                T=torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0),
                image_size=torch.tensor([[h, w]], dtype=torch.float32, device=device),
                in_ndc=False,
                device=device,
            )

            # Rasterize (no global transform on mesh here)
            rs_key = (h, w)
            if rs_key not in raster_settings_cache:
                raster_settings_cache[rs_key] = RasterizationSettings(
                    image_size=(h, w),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    bin_size=0,
                )
            rasterizer = MeshRasterizer(cameras=cameras_p3d, raster_settings=raster_settings_cache[rs_key])
            fragments = rasterizer(meshes)

            # faces_per_pixel=1 -> (1,H,W,1) -> (H,W)
            zbuf = fragments.zbuf[0, ..., 0].detach().cpu().numpy()
            zbuf = np.nan_to_num(zbuf, nan=0.0)

            depths[cam_name].append(zbuf.astype(np.float32))

    # Stack per-camera data
    cam_names = sorted(rgbs.keys(), key=lambda s: int(s))
    for cam_name in cam_names:
        if len(rgbs[cam_name]) == 0:
            # Camera had no valid frames (skip)
            del intrs[cam_name], extrs[cam_name], rgbs[cam_name], depths[cam_name]
            continue
        rgbs[cam_name] = np.stack(rgbs[cam_name]).transpose(0, 3, 1, 2).astype(np.uint8)  # [T,3,H,W]
        depths[cam_name] = np.stack(depths[cam_name]).astype(np.float32)  # [T,H,W]

    # Basic shape checks (use first cam as reference)
    kept_cams = sorted(rgbs.keys(), key=lambda s: int(s))
    assert len(kept_cams) > 0, "No cameras with data."
    n_frames, _, h, w = rgbs[kept_cams[0]].shape
    for cam_name in kept_cams:
        assert rgbs[cam_name].shape == (n_frames, 3, h, w)
        assert intrs[cam_name].shape == (3, 3)
        assert extrs[cam_name].shape == (3, 4)
        assert depths[cam_name].shape == (n_frames, h, w)

    # Rotate the scene to have the ground at z=0
    rot_x = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    rot_y = Rotation.from_euler('y', 0, degrees=True).as_matrix()
    rot_z = Rotation.from_euler('z', 0, degrees=True).as_matrix()
    rot = torch.from_numpy(rot_z @ rot_y @ rot_x)
    translation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    for cam_name in kept_cams:
        E = torch.from_numpy(extrs[cam_name][None, None])  # [1,1,3,4]
        E_tx = transform_scene(1, rot, translation, None, E, None, None, None)[1]
        extrs[cam_name] = E_tx[0, 0].numpy()

    # Save
    save_pickle(save_pkl_path, dict(
        rgbs=rgbs,
        intrs=intrs,
        extrs=extrs,
        depths=depths,
        ego_cam_name=None,
    ))

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

        mesh_vertices = meshes._verts_list[0].cpu()
        mesh_faces = meshes._faces_list[0].cpu()
        mesh_vertices = transform_scene(1, rot, translation, None, None, None, mesh_vertices[None], None)[3][0]
        rr.log(
            "mesh",
            rr.Mesh3D(
                vertex_positions=mesh_vertices.numpy().astype(np.float32),  # (N, 3)
                triangle_indices=mesh_faces.numpy().reshape(-1, 3).astype(np.int32),  # (M, 3)
                albedo_factor=[200, 200, 255],  # Optional color
            ),
        )

        fps = 30
        for frame_idx in range(n_frames):
            rr.set_time_seconds("frame", frame_idx / fps)
            for cam_name in cam_names:
                extr = extrs[cam_name]
                intr = intrs[cam_name]
                img = rgbs[cam_name][frame_idx].transpose(1, 2, 0).astype(np.uint8)
                depth = depths[cam_name][frame_idx]

                h, w = img.shape[:2]
                fx, fy = intr[0, 0], intr[1, 1]
                cx, cy = intr[0, 2], intr[1, 2]

                # Camera pose
                T = np.eye(4)
                T[:3, :] = extr
                world_T_cam = np.linalg.inv(T)
                rr.log(f"{cam_name}/image", rr.Transform3D(
                    translation=world_T_cam[:3, 3],
                    mat3x3=world_T_cam[:3, :3],
                ))
                rr.log(f"{cam_name}/image", rr.Pinhole(
                    image_from_camera=intr,
                    width=w,
                    height=h
                ))
                rr.log(f"{cam_name}/image", rr.Image(img))

                rr.log(f"{cam_name}/depth", rr.Transform3D(
                    translation=world_T_cam[:3, 3],
                    mat3x3=world_T_cam[:3, :3],
                ))
                rr.log(f"{cam_name}/depth", rr.Pinhole(
                    image_from_camera=intr,
                    width=w,
                    height=h
                ))
                rr.log(f"{cam_name}/depth", rr.DepthImage(depth, meter=1.0, colormap="viridis"))

                # Unproject depth to point cloud
                y, x = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
                z = depth
                valid = z > 0
                x = x[valid]
                y = y[valid]
                z = z[valid]

                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                pts_cam = np.stack([X, Y, z], axis=-1)

                # Transform to world
                R = world_T_cam[:3, :3]
                t = world_T_cam[:3, 3]
                pts_world = pts_cam @ R.T + t

                # Color
                colors = img[y, x]

                rr.log(f"point_cloud/{cam_name}", rr.Points3D(positions=pts_world, colors=colors))

        if rerun_mode == "save":
            base, name = os.path.split(save_pkl_path)
            name_no_ext = os.path.splitext(name)[0]
            save_rrd_path = os.path.join(base, f"rerun__{name_no_ext}.rrd")
            rr.save(save_rrd_path)
            print(f"Saved rerun viz to {os.path.abspath(save_rrd_path)}")

    print(f"Done with {save_pkl_path}.")
    print()


if __name__ == "__main__":
    dataset_root = "datasets/hi4d"
    output_root = "datasets/hi4d-processed"

    longside_resolution: Optional[int] = 512
    if longside_resolution is not None:
        output_root += f"-resized-{longside_resolution}"
    os.makedirs(output_root, exist_ok=True)

    pairs = [
        "pair00", "pair01", "pair02", "pair09", "pair10",
        "pair12", "pair13", "pair14", "pair15", "pair16",
        "pair17", "pair18", "pair19", "pair21", "pair22",
        "pair23", "pair27", "pair28", "pair32", "pair37"
    ]

    # Enumerate actions per pair automatically
    for pair in tqdm.tqdm(pairs, desc="Pairs"):
        pair_dir = os.path.join(dataset_root, pair)
        assert os.path.isdir(pair_dir)
        actions = sorted([
            d for d in os.listdir(pair_dir)
            if os.path.isdir(os.path.join(pair_dir, d)) and not d.startswith(".")
        ])

        for action in tqdm.tqdm(actions, desc=f"{pair} actions", leave=False):
            out_pkl = os.path.join(output_root, f"{pair}__{action}.pkl")
            extract_hi4d_action_to_pkl(
                dataset_root=dataset_root,
                pair=pair,
                action=action,
                save_pkl_path=out_pkl,
                downscaled_longerside=longside_resolution,
                save_rerun_viz=True,
                stream_rerun_viz=False,
                skip_if_output_exists=True,
            )
