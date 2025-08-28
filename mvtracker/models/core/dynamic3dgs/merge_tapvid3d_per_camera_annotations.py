import json
import os
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import rerun as rr
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm

from .helpers import setup_camera
from .visualize import log_tracks_to_rerun


def to_homogeneous(x):
    return np.concatenate([x, np.ones_like(x[..., :1])], axis=-1)


def from_homogeneous(x, assert_homogeneous_part_is_equal_to_1=False, eps=0.001):
    if assert_homogeneous_part_is_equal_to_1:
        assert np.allclose(x[..., -1:], 1, atol=eps), f"Expected homogeneous part to be 1, got {x[..., -1:]}"
    return x[..., :-1] / x[..., -1:]


def load_scene_data(params_path, seg_as_col=False):
    """Load 3D scene data from file."""
    params = dict(np.load(params_path, allow_pickle=True))
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    is_fg = params['seg_colors'][:, 0] > 0.5
    scene_data = []
    for t in range(len(params['means3D'])):
        rendervar = {
            'means3D': params['means3D'][t],
            'colors_precomp': params['rgb_colors'][t] if not seg_as_col else params['seg_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations'][t]),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(params['log_scales']),
            'means2D': torch.zeros_like(params['means3D'][0], device="cuda")
        }
        scene_data.append(rendervar)
    return scene_data, is_fg


def render(h, w, k, w2c, timestep_data, near=0.01, far=100.0):
    """Render scene using Gaussian Rasterization."""
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def merge_annotations(
        scene_root,
        checkpoint_path,
        tapvid3d_annotation_paths,
        nearest_neighbor_distance_threshold_for_visibility=0.015,
        skip_if_output_already_exists=False,

        assert_query_points_project_to_trajectories_in_tapvid3d_annotation=False,

        rerun_logging=False,
        rerun_stream_only=False,
        rerun_views_to_viz=(27, 16, 1),

        rerun_log_rgb=True,
        rerun_log_d3dgs_rgb=False,
        rerun_log_d3dgs_depth=False,
        rerun_log_d3dgs_point_cloud=True,
        rerun_log_tracks=True,
        rerun_log_n_skip_t=1,
):
    output_annotation_path = scene_root / "tapvid3d_annotations.npz"
    if skip_if_output_already_exists and output_annotation_path.exists():
        print(f"Output file {output_annotation_path} already exists, skipping.")
        return

    scene_data, is_fg = load_scene_data(os.path.join(checkpoint_path, "params.npz"))
    md_train = json.load(open(os.path.join(scene_root, "train_meta.json"), "r"))
    md_test = json.load(open(os.path.join(scene_root, "test_meta.json"), "r"))

    views = sorted(list(set(md_train["cam_id"][0]) | set(md_test["cam_id"][0])))
    assert list(range(31)) == views, "We expect exactly 31 views: from 0 to 30."
    n_frames = len(md_train['fn'])
    n_views = len(views)

    # Check that the selected views are in the training set
    view_paths = []
    for view_idx in views:
        view_path = scene_root / "ims" / f"{view_idx}"
        assert view_path.exists()
        view_paths.append(view_path)
    frame_paths = [sorted(view_path.glob("*.jpg")) for view_path in view_paths]
    assert all(len(frame_paths[v]) == n_frames for v in range(n_views))
    assert len(scene_data) == n_frames

    # Load the camera parameters
    fx, fy, cx, cy, extrinsics = [], [], [], [], []
    for view_idx in views:
        fx_current, fy_current, cx_current, cy_current, extrinsics_current = [], [], [], [], []
        for t in range(n_frames):
            if view_idx in md_train['cam_id'][t]:
                md = md_train
            elif view_idx in md_test['cam_id'][t]:
                md = md_test
            else:
                raise ValueError(f"Camera {view_idx} not found in any of the meta files")

            view_idx_in_array = md['cam_id'][t].index(view_idx)
            k = md['k'][t][view_idx_in_array]
            w2c = np.array(md['w2c'][t][view_idx_in_array])

            fx_current.append(k[0][0])
            fy_current.append(k[1][1])
            cx_current.append(k[0][2])
            cy_current.append(k[1][2])
            extrinsics_current.append(w2c)

        assert all(np.equal(fx_current[0], fx_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(fy_current[0], fy_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(cx_current[0], cx_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(cy_current[0], cy_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(extrinsics_current[0], extrinsics_current[t]).all() for t in range(1, n_frames))

        fx.append(fx_current[0])
        fy.append(fy_current[0])
        cx.append(cx_current[0])
        cy.append(cy_current[0])
        extrinsics.append(extrinsics_current[0])

    k = np.eye(3).astype(np.float64)[None].repeat(n_views, 0)
    k[:, 0, 0] = fx
    k[:, 1, 1] = fy
    k[:, 0, 2] = cx
    k[:, 1, 2] = cy
    extrinsics = np.stack(extrinsics).astype(np.float64)
    k_inv = np.linalg.inv(k)
    extrinsics_inv = np.linalg.inv(extrinsics)

    # Render imgs and depths
    rgbs = np.stack([
        np.stack([
            np.array(Image.open(frame_paths[v][t]))
            for t in range(n_frames)
        ])
        for v in range(n_views)
    ])

    h, w = rgbs.shape[2], rgbs.shape[3]
    d3dgs_rgbs = []
    d3dgs_depths = []
    for v, view_idx in enumerate(views):
        for t in range(n_frames):
            im, depth = render(h, w, k[v], extrinsics[v], scene_data[t])
            d3dgs_rgbs.append(im.cpu().numpy().transpose(1, 2, 0))
            d3dgs_depths.append(depth.cpu().numpy()[0])
    d3dgs_rgbs = np.stack(d3dgs_rgbs).reshape(n_views, n_frames, h, w, 3)
    d3dgs_depths = np.stack(d3dgs_depths).reshape(n_views, n_frames, h, w)

    assert rgbs.shape == (n_views, n_frames, h, w, 3)
    assert d3dgs_rgbs.shape == (n_views, n_frames, h, w, 3)
    assert d3dgs_depths.shape == (n_views, n_frames, h, w)

    # Merge TAP-Vid3D annotations
    merged_trajectories = []
    merged_trajectories_pixelspace = []
    merged_per_view_visibilities = []
    merged_query_points_3d = []
    for tapvid3d_annotation_path in tqdm(tapvid3d_annotation_paths):
        annotation = np.load(tapvid3d_annotation_path)
        queries_xyt = annotation["queries_xyt"]
        tracks_XYZ = annotation["tracks_XYZ"]
        visibility = annotation["visibility"]
        fx_fy_cx_cy = annotation["fx_fy_cx_cy"]
        images_jpeg_bytes = annotation["images_jpeg_bytes"]

        _, cam_id = os.path.basename(tapvid3d_annotation_path)[:-4].split("_")
        cam_id = int(cam_id)
        assert cam_id == views.index(cam_id)

        n_tracks, _ = queries_xyt.shape
        assert cam_id in views
        assert queries_xyt.shape == (n_tracks, 3)
        assert fx_fy_cx_cy.shape == (4,)
        assert images_jpeg_bytes.shape == (n_frames,)
        assert tracks_XYZ.shape == (n_frames, n_tracks, 3)
        assert visibility.shape == (n_frames, n_tracks)
        assert np.allclose(fx_fy_cx_cy, [fx[cam_id], fy[cam_id], cx[cam_id], cy[cam_id]])

        # Project the tracks to the world space
        cam_coords_homo = to_homogeneous(tracks_XYZ)
        world_coords_homo = np.einsum("ij,SNj->SNi", extrinsics_inv[cam_id], cam_coords_homo)
        world_coords = from_homogeneous(world_coords_homo, assert_homogeneous_part_is_equal_to_1=True)

        # Project query points to 3D to verify we can reproduce the camera space points
        qp_t = queries_xyt[:, 2].astype(np.int32)
        qp_xy_pixel = queries_xyt[:, :2].astype(np.float32)
        qp_depth = np.ones((n_tracks, 1), dtype=np.float32) * np.inf
        qp_xyz_camera = np.ones((n_tracks, 3), dtype=np.float32) * np.inf
        qp_xyz_world = np.ones((n_tracks, 3), dtype=np.float32) * np.inf
        for t in range(n_frames):
            qp_mask = qp_t == t
            if qp_mask.sum() == 0:
                continue

            # V2 depth interpolation
            x_nearest = qp_xy_pixel[qp_mask, 0].round().astype(np.int32).clip(0, w - 1)
            y_nearest = qp_xy_pixel[qp_mask, 1].round().astype(np.int32).clip(0, h - 1)
            depth_nearest = d3dgs_depths[cam_id, t].reshape(-1)[
                (y_nearest * w + x_nearest).reshape(-1)]
            depth_nearest = depth_nearest.reshape(-1, 1)
            qp_depth[qp_mask] = depth_nearest

            qp_xyz_pixel_t = np.concatenate([qp_xy_pixel[qp_mask], np.ones_like(qp_xy_pixel[qp_mask][..., :1])],
                                            axis=1)
            qp_xyz_camera_t = np.einsum("ij,Nj->Ni", k_inv[cam_id], qp_xyz_pixel_t) * qp_depth[qp_mask]
            qp_xyz_world_t = np.einsum("ij,Nj->Ni", extrinsics_inv[cam_id],
                                       np.concatenate([qp_xyz_camera_t, np.ones_like(qp_xyz_camera_t[..., :1])],
                                                      axis=1))[:, :3]

            qp_xyz_camera[qp_mask] = qp_xyz_camera_t
            qp_xyz_world[qp_mask] = qp_xyz_world_t

        assert np.all(np.isfinite(qp_depth))
        assert np.all(np.isfinite(qp_xyz_camera))
        assert np.all(np.isfinite(qp_xyz_world))

        # Verify that the query points are close to the tracks in the world space
        qp_projection_diff = np.linalg.norm(
            qp_xyz_camera - tracks_XYZ[queries_xyt[:, 2].astype(np.int32), np.arange(n_tracks)], axis=1)
        repro1 = np.percentile(qp_projection_diff, 80) < 1
        repro2 = qp_projection_diff.mean() < 0.1
        if not repro1 or not repro2:
            warnings.warn(f"Projecting query points to match tracks in camera space failed. "
                          f"Differences: max={qp_projection_diff.max():0.3f}, "
                          f"mean={qp_projection_diff.mean():0.3f}, "
                          f"median={np.percentile(qp_projection_diff, 50):0.3f}, "
                          f"p80={np.percentile(qp_projection_diff, 80):0.3f}")
        if assert_query_points_project_to_trajectories_in_tapvid3d_annotation:
            assert repro1
            assert repro2

        # Verify that the projected tracks are close to the query points in pixel space
        cam_coords_per_view = from_homogeneous(np.einsum("Vij,SNj->VSNi", extrinsics, world_coords_homo), True)
        pixel_coords_per_view = from_homogeneous(np.einsum("Vij,VSNj->VSNi", k, cam_coords_per_view))
        diff = np.linalg.norm(qp_xy_pixel - pixel_coords_per_view[cam_id][qp_t, np.arange(n_tracks)], axis=-1)
        repro3 = np.percentile(diff, 80) < 0.1
        # The xy pixel query from queries_xyz in the raw labels sometimes doesn't match the tracks_XYZ in camera space.
        # In the merged labels, we will not use the queries_xyz, but just directly work with the tracks_XYZ and their
        # projections (where pixel-space projections are needed).
        if not repro3:
            warnings.warn(f"Projecting tracks to pixel space to match query points failed. "
                          f"Max diff: {diff.max()}. Mean diff: {diff.mean()}. Median diff: {np.percentile(diff, 50)}. "
                          f"Percentile 80: {np.percentile(diff, 80)}.")
        if assert_query_points_project_to_trajectories_in_tapvid3d_annotation:
            assert repro3

        # import matplotlib.pyplot as plt
        # plt.imshow(rgbs[v, qp_t[0]])
        # plt.scatter(qp_xy_pixel[0, 0], qp_xy_pixel[0, 1], color="red")
        # plt.scatter(pixel_coords_per_view[cam_id][qp_t[0], 0, 0], pixel_coords_per_view[cam_id][qp_t[0], 0, 1], color="green")
        # plt.show()

        # Compute the distance from the trajectories to their nearest depthmap neighbors
        depthmap_nearest_neighbor_distance = np.ones((n_views, n_frames, n_tracks), dtype=np.float32) * np.inf
        k_inv_torch = torch.from_numpy(k_inv).cuda()
        extrinsics_inv_torch = torch.from_numpy(extrinsics_inv).cuda()
        pixel_coords_per_view_round_torch = torch.from_numpy(pixel_coords_per_view.round().astype(int)).cuda()
        world_coords_torch = torch.from_numpy(world_coords).cuda()
        for v, view_idx in enumerate(views):
            for t in range(n_frames):
                # Project depths to world space
                # Pixel --> Camera --> World
                pixel_xy = torch.stack(torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy"), dim=-1).cuda()
                pixel_xy = pixel_xy.type(k_inv_torch.dtype)
                pixel_xy_homo = torch.cat([pixel_xy, torch.ones_like(pixel_xy[..., :1])], dim=-1)
                depthmap_camera_xyz = torch.einsum("ij,hwj->hwi", k_inv_torch[v], pixel_xy_homo)
                depthmap_camera_xyz *= torch.tensor(d3dgs_depths[v, t], device="cuda", dtype=torch.float32)[..., None]
                depthmap_camera_xyz_homo = torch.cat(
                    [depthmap_camera_xyz, torch.ones_like(depthmap_camera_xyz[..., :1])], dim=-1)
                depthmap_world_xyz_homo = torch.einsum("ij,hwj->hwi", extrinsics_inv_torch[v], depthmap_camera_xyz_homo)
                depthmap_world_xyz = depthmap_world_xyz_homo[..., :-1] / depthmap_world_xyz_homo[..., -1:]

                radius = 3
                xmin = (pixel_coords_per_view_round_torch[v, t, :, 0] - radius).clip(min=0, max=w - 1 - 2 * radius)
                ymin = (pixel_coords_per_view_round_torch[v, t, :, 1] - radius).clip(min=0, max=h - 1 - 2 * radius)
                offsets = torch.arange(0, 2 * radius + 1, device="cuda")
                x_offsets, y_offsets = torch.meshgrid(offsets, offsets, indexing="ij")
                x_offsets = x_offsets.reshape(-1)
                y_offsets = y_offsets.reshape(-1)
                x_indices = (xmin[:, None] + x_offsets[None, :]).long()
                y_indices = (ymin[:, None] + y_offsets[None, :]).long()
                neighbors = depthmap_world_xyz[y_indices, x_indices]
                nearest_dist = torch.linalg.norm(neighbors - world_coords_torch[t][:, None, :], dim=-1).min(dim=-1)[0]
                depthmap_nearest_neighbor_distance[v, t, :] = nearest_dist.cpu().numpy()
        assert not np.isinf(depthmap_nearest_neighbor_distance).any()

        # Compute whether the projected trajectory is within the HxW frame of a view
        within_frame = ((pixel_coords_per_view[..., 0] >= 0) & (pixel_coords_per_view[..., 0] < w)
                        & (pixel_coords_per_view[..., 1] >= 0) & (pixel_coords_per_view[..., 1] < h))

        # If nearest neighbor in depth is less than X cm away, consider the point as visible in that view
        # Furthermore if the projected pixel space location is out of the frame, the point is not visible
        per_view_visibility = depthmap_nearest_neighbor_distance <= nearest_neighbor_distance_threshold_for_visibility
        per_view_visibility = per_view_visibility & within_frame

        valid_tracks_mask = (per_view_visibility[cam_id] == visibility).mean(0) > 0.7
        valid_tracks_indices = np.where(valid_tracks_mask)[0]
        assert (per_view_visibility[cam_id] == visibility)[:, valid_tracks_mask].mean() > 0.8

        query_points_3d_t = np.max(np.stack([qp_t, per_view_visibility[cam_id].argmax(0)], axis=1), axis=1)
        query_points_3d_xyz = world_coords[query_points_3d_t, np.arange(n_tracks)]
        query_points_3d = np.concatenate([query_points_3d_t[:, None], query_points_3d_xyz[:, :]], axis=1)

        merged_trajectories.append(world_coords[:, valid_tracks_indices, :])
        merged_trajectories_pixelspace.append(pixel_coords_per_view[:, :, valid_tracks_indices, :])
        merged_per_view_visibilities.append(per_view_visibility[:, :, valid_tracks_indices])
        merged_query_points_3d.append(query_points_3d[valid_tracks_indices])

        # print(f"VERBOSE LOGS: varying the distance threshold for cam_id={cam_id}")
        # for d in [0.001, 0.005, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.019,
        #           0.020, 0.021, 0.022, 0.023, 0.024, 0.025, 0.026, 0.027, 0.028, 0.029, 0.030, 0.035, 0.04, 0.05]:
        #     per_view_visibility = (depthmap_nearest_neighbor_distance <= d) & within_frame
        #     print(f" --> dist={d:0.3f} "
        #           f"v1={per_view_visibility[cam_id].mean() * 100:.1f} "
        #           f"v2={visibility.mean() * 100:.1f} "
        #           f"acc={(per_view_visibility[cam_id] == visibility).mean() * 100:.1f}")
        # per_view_visibility = depthmap_nearest_neighbor_distance <= nearest_neighbor_distance_threshold_for_visibility
        # per_view_visibility = per_view_visibility & within_frame
        # print(f"dist={nearest_neighbor_distance_threshold_for_visibility:0.3f} "
        #       f"v1={per_view_visibility[cam_id].mean() * 100:.1f} "
        #       f"v2={visibility.mean() * 100:.1f} "
        #       f"acc={(per_view_visibility[cam_id] == visibility).mean() * 100:.1f}")
        #
        # if cam_id != 16:
        #     continue
        #
        # rr.init("reconstruction", recording_id="v0.1")
        # rr.connect_tcp()
        # rr.log("/", rr.ViewCoordinates.LEFT_HAND_Y_DOWN, static=True)
        # rr.set_time_seconds("frame", 0)
        # rr.log("world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        #                                 colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
        #
        # rr.log(f"debug/qp_xyz_camera",
        #        rr.Points3D(world_coords[queries_xyt[:, 2].astype(np.int32), np.arange(n_tracks)],
        #                    colors=np.ones_like(qp_xyz_camera) * [0, 1, 0], radii=0.01))
        # rr.log(f"debug/qp_xyz_camera_reproj",
        #        rr.Points3D(qp_xyz_world, colors=np.ones_like(qp_xyz_camera) * [0, 0, 1], radii=0.01))
        # strips = np.stack([world_coords[queries_xyt[:, 2].astype(np.int32), np.arange(n_tracks)], qp_xyz_world], axis=1)
        # rr.log("debug/qp_xyz_error_line", rr.LineStrips3D(strips=strips, colors=np.array([1., 0, 0]), radii=0.003))
        #
        # seq = os.path.basename(scene_root)
        # for t in range(0, n_frames, rerun_log_n_skip_t):
        #     for v in rerun_views_to_viz:
        #         rr.set_time_seconds("frame", t / 30)
        #         depth_values = d3dgs_depths[v, t].ravel()
        #         valid_mask = depth_values > 0
        #         y, x = np.indices((h, w))
        #         homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
        #         cam_coords = (k_inv[v] @ homo_pixel_coords) * depth_values
        #         cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
        #         world_coords_ = (extrinsics_inv[v] @ cam_coords)[:3].T
        #         world_coords_ = world_coords_[valid_mask]
        #         rgb_colors = rgbs[v, t].reshape(-1, 3)[valid_mask].astype(np.uint8)
        #         rr.log(f"{seq}/dyn-3dgs-point-cloud/view-{v}",
        #                rr.Points3D(world_coords_, colors=rgb_colors, radii=0.004))
        # cmap = matplotlib.colormaps["gist_rainbow"]
        # norm = matplotlib.colors.Normalize(vmin=world_coords[..., 0].min(), vmax=world_coords[..., 0].max())
        # track_colors = cmap(norm(world_coords[-1, :, 0]))
        # log_tracks_to_rerun(
        #     tracks=world_coords,
        #     visibles=visibility,
        #     query_timestep=np.zeros(n_tracks, dtype=np.int32),
        #     colors=track_colors,
        #     track_names=[f"track-{i:02d}" for i in range(n_tracks)],
        #     entity_format_str=f"debug/tapvid3d-tracks-visGT/{{}}",
        #     invisible_color=[0.3, 0.3, 0.3],
        # )
        # log_tracks_to_rerun(
        #     tracks=world_coords,
        #     visibles=per_view_visibility[views.index(16)],
        #     query_timestep=np.zeros(n_tracks, dtype=np.int32),
        #     colors=track_colors,
        #     track_names=[f"track-{i:02d}" for i in range(n_tracks)],
        #     entity_format_str=f"debug/tapvid3d-tracks-vis16-v2/{{}}",
        #     invisible_color=[0.3, 0.3, 0.3],
        # )
        # log_tracks_to_rerun(
        #     tracks=world_coords,
        #     visibles=per_view_visibility[views.index(27)],
        #     query_timestep=np.zeros(n_tracks, dtype=np.int32),
        #     colors=track_colors,
        #     track_names=[f"track-{i:02d}" for i in range(n_tracks)],
        #     entity_format_str=f"debug/tapvid3d-tracks-vis27/{{}}",
        #     invisible_color=[0.3, 0.3, 0.3],
        # )
        # exit()
    merged_trajectories = np.concatenate(merged_trajectories, axis=1)
    merged_trajectories_pixelspace = np.concatenate(merged_trajectories_pixelspace, axis=2)
    merged_per_view_visibilities = np.concatenate(merged_per_view_visibilities, axis=2)
    merged_query_points_3d = np.concatenate(merged_query_points_3d, axis=0)

    # Remove duplicates from the merged trajectories
    from sklearn.cluster import DBSCAN
    flat_trajectories = merged_trajectories.transpose(1, 0, 2).reshape(-1, n_frames * 3)
    dbscan = DBSCAN(eps=0.01, min_samples=1, metric='euclidean')
    labels = dbscan.fit_predict(flat_trajectories)
    _, unique_indices = np.unique(labels, return_index=True)
    unique_indices = np.sort(unique_indices)
    merged_trajectories = merged_trajectories[:, unique_indices, :]
    merged_trajectories_pixelspace = merged_trajectories_pixelspace[:, :, unique_indices, :]
    merged_per_view_visibilities = merged_per_view_visibilities[:, :, unique_indices]
    merged_query_points_3d = merged_query_points_3d[unique_indices, :]

    n_tracks = merged_trajectories.shape[1]
    assert merged_trajectories.shape == (n_frames, n_tracks, 3)
    assert merged_trajectories_pixelspace.shape == (n_views, n_frames, n_tracks, 2)
    assert merged_per_view_visibilities.shape == (n_views, n_frames, n_tracks)
    assert merged_query_points_3d.shape == (n_tracks, 4)

    # Shuffle the tracks
    np.random.seed(72)
    track_perm = np.random.permutation(n_tracks)
    shuffled_trajectories = merged_trajectories[:, track_perm, :]
    shuffled_trajectories_pixelspace = merged_trajectories_pixelspace[:, :, track_perm, :]
    shuffled_per_view_visibilities = merged_per_view_visibilities[:, :, track_perm]
    shuffled_query_points_3d = merged_query_points_3d[track_perm, :]

    # Save the merged annotations
    np.savez(
        output_annotation_path,
        trajectories=shuffled_trajectories,
        trajectories_pixelspace=shuffled_trajectories_pixelspace,
        per_view_visibilities=shuffled_per_view_visibilities,
        query_points_3d=shuffled_query_points_3d,
        intrinsics=k,
        extrinsics=extrinsics,
    )
    print(f"Saved merged annotations to {output_annotation_path}")

    if rerun_logging:
        rr.init("reconstruction", recording_id="v0.1")
        if rerun_stream_only:
            rr.connect_tcp()
        rr.set_time_seconds("frame", 0)
        rr.log("/", rr.ViewCoordinates.LEFT_HAND_Y_DOWN, static=True)
        rr.log("world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                                        colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

        seq = os.path.basename(scene_root)
        for t in range(0, n_frames, rerun_log_n_skip_t):
            for v in rerun_views_to_viz:
                rr.set_time_seconds("frame", t / 30)
                if rerun_log_rgb:
                    rr.log(f"{seq}/rgb/view-{views[v]}/rgb", rr.Image(rgbs[v, t]))
                    rr.log(f"{seq}/rgb/view-{views[v]}", rr.Pinhole(image_from_camera=k[v], width=w, height=h))
                    rr.log(f"{seq}/rgb/view-{views[v]}", rr.Transform3D(translation=extrinsics_inv[v, :3, 3],
                                                                        mat3x3=extrinsics_inv[v, :3, :3]))
                if rerun_log_d3dgs_rgb:
                    rr.log(f"{seq}/dyn-3dgs-rgb/view-{views[v]}/rgb", rr.Image(d3dgs_rgbs[v, t]))
                    rr.log(f"{seq}/dyn-3dgs-rgb/view-{views[v]}", rr.Pinhole(image_from_camera=k[v], width=w, height=h))
                    rr.log(f"{seq}/dyn-3dgs-rgb/view-{views[v]}", rr.Transform3D(translation=extrinsics_inv[v, :3, 3],
                                                                                 mat3x3=extrinsics_inv[v, :3, :3]))
                if rerun_log_d3dgs_depth:
                    rr.log(f"{seq}/dyn-3dgs-depth/view-{views[v]}/depth",
                           rr.DepthImage(d3dgs_depths[v, t], point_fill_ratio=0.2))
                    rr.log(f"{seq}/dyn-3dgs-depth/view-{views[v]}",
                           rr.Pinhole(image_from_camera=k[v], width=w, height=h))
                    rr.log(f"{seq}/dyn-3dgs-depth/view-{views[v]}",
                           rr.Transform3D(translation=extrinsics_inv[v, :3, 3], mat3x3=extrinsics_inv[v, :3, :3]))

                if rerun_log_d3dgs_point_cloud:
                    depth_values = d3dgs_depths[v, t].ravel()
                    valid_mask = depth_values > 0
                    y, x = np.indices((h, w))
                    homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
                    cam_coords = (k_inv[v] @ homo_pixel_coords) * depth_values
                    cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                    world_coords = (extrinsics_inv[v] @ cam_coords)[:3].T
                    world_coords = world_coords[valid_mask]
                    rgb_colors = rgbs[v, t].reshape(-1, 3)[valid_mask].astype(np.uint8)
                    rr.log(f"{seq}/dyn-3dgs-point-cloud/view-{v}",
                           rr.Points3D(world_coords, colors=rgb_colors, radii=0.004))

        if rerun_log_tracks:
            raw_tracks = np.stack([data['means3D'][is_fg][::200].contiguous().cpu().numpy() for data in scene_data])
            n_tracks_raw = raw_tracks.shape[1]
            cmap = matplotlib.colormaps["gist_rainbow"]
            norm = matplotlib.colors.Normalize(vmin=raw_tracks[..., 0].min(), vmax=raw_tracks[..., 0].max())
            track_colors = cmap(norm(raw_tracks[-1, :, 0]))
            log_tracks_to_rerun(
                tracks=raw_tracks,
                visibles=np.ones((n_frames, n_tracks_raw), dtype=bool),
                query_timestep=np.zeros(n_tracks_raw, dtype=np.int32),
                colors=track_colors,
                track_names=[f"track-{i:02d}" for i in range(n_tracks_raw)],
                entity_format_str=f"{seq}/dyn-3dgs-raw-tracks/{{}}",
                invisible_color=[0.3, 0.3, 0.3],
            )

            cmap = matplotlib.colormaps["gist_rainbow"]
            norm = matplotlib.colors.Normalize(vmin=shuffled_trajectories[..., 0].min(),
                                               vmax=shuffled_trajectories[..., 0].max())
            track_colors = cmap(norm(shuffled_trajectories[-1, :, 0]))
            batch_size = 50
            max_tracks = 500
            for v in rerun_views_to_viz:
                for tracks_batch_start in range(0, max_tracks, batch_size):
                    tracks_batch_end = min(tracks_batch_start + batch_size, n_tracks)
                    log_tracks_to_rerun(
                        tracks=shuffled_trajectories[:, tracks_batch_start:tracks_batch_end],
                        visibles=shuffled_per_view_visibilities[v, :, tracks_batch_start:tracks_batch_end],
                        query_timestep=shuffled_query_points_3d[:, 0][tracks_batch_start:tracks_batch_end].astype(int),
                        colors=track_colors[tracks_batch_start:tracks_batch_end],
                        track_names=[f"track-{i:02d}" for i in range(tracks_batch_start, tracks_batch_end)],
                        entity_format_str=f"{seq}/tapvid3d-tracks/view-{v}-visiblity/{tracks_batch_start}-{tracks_batch_end}/{{}}",
                        invisible_color=[0.3, 0.3, 0.3],
                    )

        if not rerun_stream_only:
            rr_rrd_path = scene_root / "rerun_tapvid3d_labels.rrd"
            rr.save(rr_rrd_path)
            print(f"Saved Rerun recording to: {os.path.abspath(rr_rrd_path)}")


if __name__ == "__main__":
    print("Merging TAP-Vid3D per-camera annotations.")
    for sequence_name in tqdm(["basketball", "boxes", "football", "juggle", "softball", "tennis"]):
        scene_root = Path(f"./datasets/panoptic_d3dgs/{sequence_name}")
        checkpoint_path = Path(f"./dynamic3dgs/output/pretrained/{sequence_name}")
        tapvid3d_annotation_paths = list(Path(f"./datasets/tapvid3d_dataset/pstudio").glob(f"{sequence_name}_*.npz"))
        merge_annotations(
            scene_root,
            checkpoint_path,
            tapvid3d_annotation_paths,
            skip_if_output_already_exists=True,
            rerun_logging=True
        )
