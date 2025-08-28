import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from external import build_rotation

REMOVE_BACKGROUND = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

w, h = 512, 512
near, far = 0.01, 100.0

from mvtracker.evaluation.evaluator_3dpt import evaluate_3dpt


def load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v, device=device).float() for k, v in params.items()}

    is_fg = params["seg_colors"][:, 0] > 0.5
    scene_data = []
    for t in range(len(params["means3D"])):
        rendervar = {
            "means3D": params["means3D"][t],
            "colors_precomp": params["rgb_colors"][t]
            if not seg_as_col
            else params["seg_colors"],
            "rotations": params["unnorm_rotations"][t],
            "opacities": torch.sigmoid(params["logit_opacities"]),
            "scales": torch.exp(params["log_scales"]),
            "means2D": torch.zeros_like(params["means3D"][0], device=device),
        }

        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def load_depth_maps(dataset_path, seq, cam_ids):
    depth_maps = {}
    for cam_id in cam_ids:
        depth_dir = f"{dataset_path}/{seq}/depths/{cam_id}/"
        depth_maps[cam_id] = []
        for frame_idx in sorted(os.listdir(depth_dir)):
            depth_path = os.path.join(depth_dir, frame_idx)
            depth_map = (
                    cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
            )
            depth_maps[cam_id].append(torch.tensor(depth_map, device=device))
        depth_maps[cam_id] = torch.stack(depth_maps[cam_id])
    return depth_maps


def preload_camera_data(dataset_path, seq, cam_ids):
    cam_params_path = f"{dataset_path}/{seq}/metadata.json"
    with open(cam_params_path, "r") as f:
        cam_params = json.load(f)

    preloaded_cameras = {}
    for cam_id in cam_ids:
        for timestamp, cameras in cam_params.items():
            if str(cam_id) in cameras:
                preloaded_cameras[cam_id] = (
                    torch.tensor(
                        cameras[str(cam_id)]["w2c"], dtype=torch.float32
                    ).cuda(),
                    torch.tensor(cameras[str(cam_id)]["k"], dtype=torch.float32).cuda(),
                )
                break  # We only need one instance per camera
    return preloaded_cameras


def gaussian_influence(point, gaussians):
    """
    Computes the most influential Gaussian for a given 3D point.

    Args:
        point (torch.Tensor): 3D point (shape: [3]).
        gaussians (dict): Dictionary containing:
            - "means3D": [N, 3] Gaussian means.
            - "scales": [N, 3] Gaussian scales.
            - "opacities": [N, 1] Gaussian opacities.
            - "rotations": [N, 4] Gaussian quaternion rotations.

    Returns:
        int: Index of the most influential Gaussian.
    """
    # print(f"Query point: {point}")

    means = gaussians["means3D"]  # [N, 3]
    scales = gaussians["scales"]  # [N, 3]
    opacities = gaussians["opacities"]  # [N, 1]
    rotations = gaussians["rotations"]  # [N, 4]

    sigmoid_opacities = opacities.squeeze()

    diff = point - means  # [N, 3]

    R = build_rotation(rotations)  # [N, 3, 3]

    S = torch.diag_embed(scales)  # [N, 3, 3]
    cov = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)  # [N, 3, 3]

    try:
        cov_inv = torch.inverse(cov)  # [N, 3, 3]
        diff = diff.unsqueeze(1)  # [N, 1, 3]
        # -1/2 * (x - mu)^T * cov^-1 * (x - mu)
        mahalanobis = (
                -0.5
                * torch.matmul(
            diff, torch.matmul(cov_inv, diff.transpose(-1, -2))
        ).squeeze()
        )  # [N]

        # Gaussian influences
        influences = sigmoid_opacities * torch.exp(mahalanobis)  # [N]

        most_influential_idx = torch.argmax(influences).item()

        print("Most influnce:", influences[most_influential_idx])

        return most_influential_idx, influences[most_influential_idx]

    except RuntimeError as e:
        print(f"Error in  computation: {e}")
        return -1


def get_visibilities(
        point_3d,
        cam_ids,
        t,
        depth_maps,
        preloaded_cameras,
        th=0.02,
):
    visibilities = []
    for cam_id in cam_ids:
        if cam_id not in preloaded_cameras:
            continue

        w2c, intrinsics = preloaded_cameras[cam_id]
        point_cam = torch.matmul(
            w2c, torch.cat([point_3d, torch.tensor([1.0], device=point_3d.device)])
        )[:3]
        X, Y, Z = point_cam
        if Z <= 0:
            continue

        x = int((X * intrinsics[0, 0]) / Z + intrinsics[0, 2])
        y = int((Y * intrinsics[1, 1]) / Z + intrinsics[1, 2])
        if not (
                0 <= x < depth_maps[cam_id].shape[2]
                and 0 <= y < depth_maps[cam_id].shape[1]
        ):
            continue

        depth_at_pixel = depth_maps[cam_id][t, y, x]
        depth_diff = Z - depth_at_pixel
        visibilities.append(0 <= depth_diff <= th)
    return visibilities


def track_query_point(
        scene_data,
        query_point,
        cam_ids,
        t_given,
        depth_maps,
        preloaded_cameras,
        threshold=0.02,
):
    """
    Tracks the 3D trajectory of a 3D query point across all frames.

    Args:
        scene_data (list): Scene data for all frames.
        query_point (tuple): Initial 2D query point (x, y).
        intrinsics (torch.Tensor): Camera intrinsics.
        t_start (int): Starting frame index.

    Returns:
        list: A list of 3D points (numpy arrays) across all timestamps.
    """
    trajectory = []
    visibilities = []

    gaussians = scene_data[t_given]
    gaussian_idx, influence = gaussian_influence(query_point, gaussians)

    for t in range(0, len(scene_data)):
        gaussians = scene_data[t]
        gaussian = {k: v[gaussian_idx] for k, v in gaussians.items()}
        point_3d_gaussian = gaussian["means3D"]
        trajectory.append(point_3d_gaussian)
        visibility = get_visibilities(
            point_3d_gaussian, cam_ids, t, depth_maps, preloaded_cameras, threshold
        )
        visibilities.append(torch.tensor(visibility))

    # print ratio of visibilities for each cam: visibity has shape n_frames * cam
    # print("Visibility ratio for each camera:")
    # print(np.array(visibility).sum(axis=0) / len(visibility))
    return trajectory, visibilities


if __name__ == "__main__":
    exp = "exp_use_duster_views_0123"
    sequences = [
        "20200709-subject-01__20200709_141754",
        "20200813-subject-02__20200813_145653",
        "20200903-subject-04__20200903_104428",
        "20200820-subject-03__20200820_135841",
        "20200908-subject-05__20200908_144409",
        "20200918-subject-06__20200918_114117",
        "20200928-subject-07__20200928_144906",
        "20201002-subject-08__20201002_110227",
        "20201015-subject-09__20201015_144721",
        "20201022-subject-10__20201022_112651",
    ]
    dataset_path = "./datasets/dex_formatted/neus_nsubsample-3"
    remove_hand = False
    use_duster = True
    cleaned_duster = False
    views = "0123"
    tracks_path = f"seed-000072_remove-hand-{remove_hand}_tracks-384_use-duster-depths-{use_duster}_clean-duster-depths-{cleaned_duster}_views-{views}_duster-views-{views}.npz"
    # sequences = ["basketball"]
    # cam_ids = [27, 16, 14, 8]
    # cam_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    cam_ids = [0, 1, 2, 3]
    for seq in sequences:
        merged_path = f"{dataset_path}/{seq}/{tracks_path}"
        # Load scene data
        scene_data, is_fg = load_scene_data(seq, exp, s=1)
        # scene_data = []
        print("Scene data loaded.")
        depth_maps = load_depth_maps(dataset_path, seq, cam_ids)
        preloaded_cameras = preload_camera_data(dataset_path, seq, cam_ids)

        load_tapvid3d = np.load(merged_path)
        query_points = load_tapvid3d["query_points_3d"]
        predictions_file = f"./output/{exp}/{seq}/predictions.npz"

        if True:
            THRESHOLD = 0.02
            predictions = []
            visibilities = []
            for i, query_point in tqdm(enumerate(query_points), desc="Query points"):
                # print("Query point:", query_point)
                given_time = query_point[0]
                # to int
                # query_point = query_point.astype(int)
                given_time = int(given_time)
                qp = query_point[1:]
                # convert it to Torch tensor
                # torch.tensor([X, Y, Z], dtype=torch.float32).cuda()
                qp = torch.tensor(qp, dtype=torch.float32).cuda()
                trajectory, visiblity_d = track_query_point(
                    scene_data,
                    qp,
                    cam_ids,
                    given_time,
                    depth_maps,
                    preloaded_cameras,
                    THRESHOLD,
                )
                # trajectory = trajectory.cpu().numpy()
                predictions.append(torch.stack(trajectory).cpu().numpy())
                visibilities.append(torch.stack(visiblity_d).cpu().numpy())

            # pred shape is: n_queries, n_frames, 3
            # convert it to n_frames, n_queries, 3
            predictions = np.array(predictions)
            predictions = np.transpose(predictions, (1, 0, 2))

            visibilities = np.array(visibilities)
            visibilities = np.transpose(visibilities, (2, 1, 0))

            preds_file = f"./output/{exp}/{seq}/predictions_threshold_{THRESHOLD}.npz"
            np.savez(
                preds_file,
                predictions=predictions,
                visibilities=visibilities,
            )
            print(f"Results saved for threshold {THRESHOLD} at: {preds_file}")

        # Load the ground truth
        query_points = load_tapvid3d["query_points_3d"]
        query_points = query_points[None, ...]  # batch * num tracks * 4
        gt_visibilities = load_tapvid3d["per_view_visibilities"]
        gt_visibilities = gt_visibilities[
            None, ...
        ]  # batch * view * num frames * num tracks
        # convert all of them to false
        gt_tracks = load_tapvid3d["trajectories"]
        gt_tracks = gt_tracks[None, ...]  # batch * num frames * num tracks * 3
        # pred_visibilities = visibilities[None, ...]
        # pred_visibilities_t = visibilities_i[None, ...]
        pred_tracks = predictions[None, ...]
        # print all dimensions for debugging
        print("query_points:", query_points.shape)
        print("gt_occluded:", gt_visibilities.shape)
        print("gt_tracks:", gt_tracks.shape)
        print("pred_occluded:", gt_visibilities.shape)
        print("pred_tracks:", pred_tracks.shape)

        gt_visibilities_any_view = gt_visibilities.any(axis=1)

        pred_visibilities = visibilities[None, ...]
        pred_visibilities_any_view = pred_visibilities.any(axis=1)
        print("EXP: ", exp)
        print("SEQ: ", seq)
        print("Evaluating ... ")
        metrics_2 = evaluate_3dpt(
            gt_tracks[0],
            gt_visibilities_any_view[0],
            pred_tracks[0],
            pred_visibilities_any_view[0],
            evaluation_setting="dex-ycb-multiview",
            query_points=query_points[0],
            track_upscaling_factor=1,
            verbose=True,
        )

        # Save evaluation results
        results_file = f"./output/{exp}/{seq}/results_threshold_{THRESHOLD}.txt"
        with open(results_file, "w") as f:
            f.write(f"Exp: {exp}\n")
            f.write(f"Seq: {seq}\n")
            f.write(f"Threshold: {THRESHOLD}\n")
            f.write(str(metrics_2))
            f.write("\n")
        print(f"Results saved at: {results_file}")

        print("Done.")
