import json
import os
from pathlib import Path

import matplotlib
import numpy as np
import rerun as rr
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from .helpers import setup_camera

RENDER_MODE = 'color'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'depth'  # 'color', 'depth' or 'centers'
# RENDER_MODE = 'centers'  # 'color', 'depth' or 'centers'

REMOVE_BACKGROUND = False  # False or True
# REMOVE_BACKGROUND = True  # False or True

FORCE_LOOP = False  # False or True
# FORCE_LOOP = True  # False or True


w, h = 640, 360
near, far = 0.01, 100.0
traj_frac = 200  # 0.5% of points
# VIEWS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
VIEWS = [1, 14]
log_rgb = True
log_d3dgs_rgb = False
log_d3dgs_depth = False
log_d3dgs_point_cloud = True
log_tracks = True
log_n_skip_view = 1
log_n_skip_t = 1


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
        if REMOVE_BACKGROUND:
            rendervar = {k: v[is_fg] for k, v in rendervar.items()}
        scene_data.append(rendervar)
    if REMOVE_BACKGROUND:
        is_fg = is_fg[is_fg]
    return scene_data, is_fg


def render(w2c, k, timestep_data):
    """Render scene using Gaussian Rasterization."""
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def log_tracks_to_rerun(
        tracks: np.ndarray,
        visibles: np.ndarray,
        query_timestep: np.ndarray,
        colors: np.ndarray,
        track_names=None,

        entity_format_str="{}",

        log_points=True,
        points_radii=0.01,
        invisible_color=[0., 0., 0.],

        log_line_strips=True,
        max_strip_length_past=30,
        max_strip_length_future=1,
        strips_radii=0.001,

        log_error_lines=False,
        error_lines_radii=0.0042,
        error_lines_color=[1., 0., 0.],
        gt_for_error_lines=None,

        fps=30,
) -> None:
    """
    Log tracks to Rerun.

    Parameters:
        tracks: Shape (T, N, 3), the 3D trajectories of points.
        visibles: Shape (T, N), boolean visibility mask for each point at each timestep.
        query_timestep: Shape (T, N), the frame index after which the tracks start.
        colors: Shape (N, 4), RGBA colors for each point.
        entity_prefix: String prefix for entity hierarchy in Rerun.
        entity_suffix: String suffix for entity hierarchy in Rerun.
    """

    T, N, _ = tracks.shape
    assert tracks.shape == (T, N, 3)
    assert visibles.shape == (T, N)
    assert query_timestep.shape == (N,)
    assert query_timestep.min() >= 0
    assert query_timestep.max() < T
    assert colors.shape == (N, 4)

    for n in range(N):
        track_name = track_names[n] if track_names is not None else f"track-{n}"
        rr.log(entity_format_str.format(track_name), rr.Clear(recursive=True))
        for t in range(query_timestep[n], T):
            rr.set_time_seconds("frame", t / fps)

            # Log the point (special handling for invisible points)
            if log_points:
                rr.log(
                    entity_format_str.format(f"{track_name}/point"),
                    rr.Points3D(
                        positions=[tracks[t, n]],
                        colors=[colors[n, :3]] if visibles[t, n] else [invisible_color],
                        radii=points_radii,
                    ),
                )

            # Log line segments for visible tracks
            if log_line_strips and t > query_timestep[n]:
                strip_t_start = max(t - max_strip_length_past, query_timestep[n].item())
                strip_t_end = min(t + max_strip_length_future, T - 1)

                strips = np.stack([
                    tracks[strip_t_start:strip_t_end, n],
                    tracks[strip_t_start + 1:strip_t_end + 1, n],
                ], axis=-2)
                strips_visibility = visibles[strip_t_start + 1:strip_t_end + 1, n]
                strips_colors = np.where(
                    strips_visibility[:, None],
                    colors[None, n, :3],
                    [invisible_color],
                )

                rr.log(
                    entity_format_str.format(f"{track_name}/line"),
                    rr.LineStrips3D(strips=strips, colors=strips_colors, radii=strips_radii),
                )

            if log_error_lines:
                assert gt_for_error_lines is not None
                strips = np.stack([
                    tracks[t, n],
                    gt_for_error_lines[t, n],
                ], axis=-2)
                rr.log(
                    entity_format_str.format(f"{track_name}/error"),
                    rr.LineStrips3D(strips=strips, colors=error_lines_color, radii=error_lines_radii),
                )


def visualize(seq, exp):
    """Visualize 3D Gaussian Splatting using Rerun."""
    scene_root = Path(f"../datasets/panoptic_d3dgs/{seq}")
    output_root = Path(f"./output/{exp}/{seq}")
    scene_data, is_fg = load_scene_data(os.path.join(output_root, "params.npz"))
    md = json.load(open(os.path.join(scene_root, "train_meta.json"), "r"))

    n_frames = len(md['fn'])
    n_views = len(VIEWS)

    # Check that the selected views are in the training set
    view_paths = []
    for view_idx in VIEWS:
        view_path = scene_root / "ims" / f"{view_idx}"
        assert view_idx in md["cam_id"][0], f"Camera {view_idx} is not in the training set"
        assert view_path.exists()
        view_paths.append(view_path)
    frame_paths = [sorted(view_path.glob("*.jpg")) for view_path in view_paths]
    assert all(len(frame_paths[v]) == n_frames for v in range(len(VIEWS)))
    assert len(scene_data) == n_frames

    # Create the output directory
    views_selection_str = '-'.join(str(v) for v in VIEWS)
    output_path = scene_root / f'dynamic3dgs-views-{views_selection_str}'
    os.makedirs(output_path, exist_ok=True)

    # Load the camera parameters
    fx, fy, cx, cy, extrinsics = [], [], [], [], []
    for view_idx in VIEWS:
        fx_current, fy_current, cx_current, cy_current, extrinsics_current = [], [], [], [], []
        for t in range(n_frames):
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

    fx = torch.tensor(fx).float()
    fy = torch.tensor(fy).float()
    cx = torch.tensor(cx).float()
    cy = torch.tensor(cy).float()
    k = torch.eye(3).float()[None].repeat(n_views, 1, 1)
    k[:, 0, 0] = fx
    k[:, 1, 1] = fy
    k[:, 0, 2] = cx
    k[:, 1, 2] = cy
    extrinsics = torch.from_numpy(np.stack(extrinsics)).float()
    k_inv = torch.inverse(k)
    extrinsics_inv = torch.inverse(extrinsics)

    # Render the depths
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
    for v, view_idx in enumerate(VIEWS):
        for t in range(n_frames):
            im, depth = render(extrinsics[v].numpy(), k[v].numpy(), scene_data[t])
            d3dgs_rgbs.append(im.cpu().numpy().transpose(1, 2, 0))
            d3dgs_depths.append(depth.cpu().numpy()[0])
    d3dgs_rgbs = np.stack(d3dgs_rgbs).reshape(n_views, n_frames, h, w, 3)
    d3dgs_depths = np.stack(d3dgs_depths).reshape(n_views, n_frames, h, w)

    assert rgbs.shape == (n_views, n_frames, h, w, 3)
    assert d3dgs_rgbs.shape == (n_views, n_frames, h, w, 3)
    assert d3dgs_depths.shape == (n_views, n_frames, h, w)

    gt_tracks = np.stack([data['means3D'][is_fg][::traj_frac].contiguous().cpu().numpy() for data in scene_data])
    n_tracks = gt_tracks.shape[1]
    gt_vis = np.ones((n_frames, n_tracks), dtype=bool)
    query_timestep = gt_vis.argmin(0)
    assert gt_tracks.shape == (n_frames, n_tracks, 3)
    assert gt_vis.shape == (n_frames, n_tracks)

    cmap = matplotlib.colormaps["gist_rainbow"]
    norm = matplotlib.colors.Normalize(vmin=gt_tracks[..., 0].min(), vmax=gt_tracks[..., 0].max())
    track_colors = cmap(norm(gt_tracks[-1, :, 0]))
    assert track_colors.shape == (n_tracks, 4)

    rr.init("reconstruction", recording_id="v0.1")
    rr.connect_tcp()
    rr.set_time_seconds("frame", 0)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                                    colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))
    for t in range(0, n_frames, log_n_skip_t):
        for v in range(0, n_views, log_n_skip_view):
            rr.set_time_seconds("frame", t / 30)
            if log_rgb:
                rr.log(f"{seq}/rgb/view-{VIEWS[v]}/rgb",
                       rr.Image(rgbs[v, t]))
                rr.log(f"{seq}/rgb/view-{VIEWS[v]}",
                       rr.Pinhole(image_from_camera=k[v].numpy(), width=w, height=h))
                rr.log(f"{seq}/rgb/view-{VIEWS[v]}",
                       rr.Transform3D(translation=extrinsics_inv[v, :3, 3].numpy(),
                                      mat3x3=extrinsics_inv[v, :3, :3].numpy()))
            if log_d3dgs_rgb:
                rr.log(f"{seq}/dyn-3dgs-rgb/view-{VIEWS[v]}/rgb",
                       rr.Image(d3dgs_rgbs[v, t]))
                rr.log(f"{seq}/dyn-3dgs-rgb/view-{VIEWS[v]}",
                       rr.Pinhole(image_from_camera=k[v].numpy(), width=w, height=h))
                rr.log(f"{seq}/dyn-3dgs-rgb/view-{VIEWS[v]}",
                       rr.Transform3D(translation=extrinsics_inv[v, :3, 3].numpy(),
                                      mat3x3=extrinsics_inv[v, :3, :3].numpy()))
            if log_d3dgs_depth:
                rr.log(f"{seq}/dyn-3dgs-depth/view-{VIEWS[v]}/depth",
                       rr.DepthImage(d3dgs_depths[v, t], point_fill_ratio=0.2))
                rr.log(f"{seq}/dyn-3dgs-depth/view-{VIEWS[v]}",
                       rr.Pinhole(image_from_camera=k[v].numpy(), width=w, height=h))
                rr.log(f"{seq}/dyn-3dgs-depth/view-{VIEWS[v]}",
                       rr.Transform3D(translation=extrinsics_inv[v, :3, 3].numpy(),
                                      mat3x3=extrinsics_inv[v, :3, :3].numpy()))
            if log_d3dgs_point_cloud:
                y, x = np.indices((h, w))
                homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
                depth_values = d3dgs_depths[v, t].ravel()
                cam_coords = (k_inv[v] @ homo_pixel_coords) * depth_values
                cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                world_coords = (extrinsics_inv[v] @ cam_coords)[:3].T
                valid_mask = depth_values > 0
                world_coords = world_coords[valid_mask]
                rgb_colors = rgbs[v, t].reshape(-1, 3)[valid_mask].astype(np.uint8)
                rr.log(f"{seq}/dyn-3dgs-point-cloud/view-{v}", rr.Points3D(world_coords, colors=rgb_colors, radii=0.01))
    if log_tracks:
        for tracks_batch_start in range(0, n_tracks, 100):
            tracks_batch_end = min(tracks_batch_start + 100, n_tracks)
            log_tracks_to_rerun(
                tracks=gt_tracks[:, tracks_batch_start:tracks_batch_end],
                visibles=gt_vis[:, tracks_batch_start:tracks_batch_end],
                query_timestep=query_timestep[tracks_batch_start:tracks_batch_end],
                colors=track_colors[tracks_batch_start:tracks_batch_end],
                track_names=[f"track-{i:02d}" for i in range(tracks_batch_start, tracks_batch_end)],
                entity_format_str=f"{seq}/dyn-3dgs-tracks/{tracks_batch_start}-{tracks_batch_end}/{{}}",
                invisible_color=[0.3, 0.3, 0.3],
            )

    print("Done with visualization.")


if __name__ == "__main__":
    exp_name = "pretrained"
    for sequence in ["basketball", "boxes", "football", "juggle", "softball", "tennis"]:
        visualize(sequence, exp_name)
