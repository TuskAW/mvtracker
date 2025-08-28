import json
import os

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm

from external import build_rotation
from helpers import setup_camera

REMOVE_BACKGROUND = False

w, h = 640, 360
near, far = 0.01, 100.0


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

        return most_influential_idx

    except RuntimeError as e:
        print(f"Error in  computation: {e}")
        return -1


def render_depth(timestep_data, w2c, k):
    """
    Renders a depth map using the Gaussian parameters.

    Args:
        timestep_data (dict): Scene data for the specific timestep.

    Returns:
        torch.Tensor: Depth map.
    """
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        (
            im,
            _,
            depth,
        ) = Renderer(raster_settings=cam)(**timestep_data)

        if depth.dim() == 3 and depth.size(0) == 1:  # Shape (1, H, W)
            depth = depth.squeeze(0)

        return depth


def load_scene_data(seq, exp, seg_as_col=False):
    params = dict(np.load(f"./output/{exp}/{seq}/params.npz"))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params["seg_colors"][:, 0] > 0.5
    scene_data = []
    for t in range(len(params["means3D"])):
        rendervar = {
            "means3D": params["means3D"][t],
            "colors_precomp": params["rgb_colors"][t]
            if not seg_as_col
            else params["seg_colors"],
            "rotations": torch.nn.functional.normalize(params["unnorm_rotations"][t]),
            "opacities": torch.sigmoid(params["logit_opacities"]),
            "scales": torch.exp(params["log_scales"]),
            "means2D": torch.zeros_like(params["means3D"][0], device="cuda"),
        }

        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return (
        scene_data,
        is_fg,
    )


def unproject_2d_to_3d(query_pt, depth_map, intrinsics):
    """
    Unproject a 2D point to 3D.
    """
    x, y = query_pt
    z = depth_map[y, x]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z

    return torch.tensor([X, Y, Z], dtype=torch.float32).cuda()


def load_camera_params(dataset_path, seq, cam_id_g):
    cam_params = f"{dataset_path}/{seq}/merged_by_timestamp.json"
    with open(cam_params, "r") as f:
        cam_params = json.load(f)
    for timestamp, cameras in cam_params.items():
        for cam_id, cam_data in cameras.items():
            if int(cam_id) == int(cam_id_g):
                return np.array(cam_data["w2c"]), np.array(cam_data["k"])
    return None, None


def c2w_convert(point_3d, w2c):
    point_3d_h = np.append(point_3d.cpu().numpy(), 1).reshape(4, 1)
    c2w = np.linalg.inv(w2c)
    point_cam = c2w @ point_3d_h
    return torch.tensor(point_cam[:3].flatten(), dtype=torch.float32).cuda()


def w2c_convert(point_3d_h, w2c):
    point_3d = np.append(point_3d_h.cpu().numpy(), 1).reshape(4, 1)
    point_cam = w2c @ point_3d
    return torch.tensor(point_cam[:3].flatten(), dtype=torch.float32).cuda()


def track_query_point(scene_data, query_point, depth_map, w2c, k, t_given=0):
    """
    Tracks the 3D trajectory of a 2D query point across all frames.

    Args:
        scene_data (list): Scene data for all frames.
        query_point (tuple): Initial 2D query point (x, y).
        intrinsics (torch.Tensor): Camera intrinsics.
        t_start (int): Starting frame index.

    Returns:
        list: A list of 3D points (numpy arrays) across all timestamps.
    """
    trajectory = []
    opacities = []
    point_3d = unproject_2d_to_3d(query_point, depth_map, k)

    point_3d_gaussian = c2w_convert(point_3d, w2c)
    gaussians = scene_data[t_given]
    gaussian_idx = gaussian_influence(point_3d_gaussian, gaussians)
    for t in range(0, len(scene_data)):
        gaussians = scene_data[t]
        gaussian = {k: v[gaussian_idx] for k, v in gaussians.items()}
        point_3d_gaussian = gaussian["means3D"]
        point_3d = w2c_convert(point_3d_gaussian, w2c)
        trajectory.append(point_3d)
        opacities.append(gaussian["opacities"])
    return trajectory


if __name__ == "__main__":
    exp = "exp_init_1-7-14-20"
    exp = "exp_merged_cleaned_pt_1-7-14-20"
    tapvid3d_dir = "./datasets/tapvid3d_dataset/pstudio"
    dataset_path = "./datasets/panoptic_d3dgs"
    # read the .npz files under directory
    npz_files = [
        f
        for f in os.listdir(tapvid3d_dir)
        if f.endswith(".npz") and "basketball" in f and "_1." in f
    ]

    file_avg_distances = {}
    # for each .npz file, it has following naming: {seq}_{cam_id}.npz
    for npz_file in tqdm(npz_files):
        seq, cam_id = npz_file.split(".")[0].split("_")

        # load tapvid3d
        gt_file = f"{tapvid3d_dir}/{npz_file}"
        print(f"Loading {gt_file}")
        data = np.load(gt_file)
        print(data.files)
        queries_xyt = data["queries_xyt"]
        print("quries_xyt:", queries_xyt)
        gt_trajectories = data["tracks_XYZ"]
        trajectories = []
        for query in tqdm(queries_xyt):
            # round to nearest integer
            q_x = round(query[0])
            q_y = round(query[1])
            query_point = (q_x, q_y)
            t_given = int(query[2]) - 1

            # Load the scene data
            scene_data, _ = load_scene_data(seq, exp)
            w2c, k = load_camera_params(dataset_path, seq, cam_id)

            depth_map = render_depth(scene_data[t_given], w2c, k)

            # Track the query point across all timestamps
            trajectory = track_query_point(
                scene_data, query_point, depth_map, w2c, k, t_given=t_given
            )

            trajectories.append(torch.stack(trajectory).cpu().numpy())

        # save the trajectories
        # np.savez(
        #     "{exp}_{seq}_{cam_id}_trajectories.npz",
        #     trajectories=trajectories.cpu().numpy(),
        # )
        # print(f"Trajectories for {seq}_{cam_id} saved.")
        distances = []
        for i, query in enumerate(queries_xyt):
            t_given = int(query[2])
            gt_traj = gt_trajectories[
                      :, i
                      ]  # Extract ground truth trajectory for this query
            exp_traj = trajectories[i]  # Our computed trajectory

            # Compute Euclidean distances for each timestamp
            per_frame_distances = np.linalg.norm(gt_traj - exp_traj, axis=1)
            avg_distance = np.mean(per_frame_distances)
            sum_distance = np.sum(per_frame_distances)
            distances.append(avg_distance)
        print(f"avg distance for {npz_file}: {np.mean(distances)}")
        file_avg_distances[npz_file] = np.mean(distances)

    print("Average distances per file:")
    print(file_avg_distances)
    print("Overall average distance:", np.mean(list(file_avg_distances.values())))
