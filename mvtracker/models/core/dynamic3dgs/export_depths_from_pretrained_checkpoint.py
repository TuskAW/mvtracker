import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm

from .helpers import setup_camera


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


def render(w, h, k, w2c, timestep_data, near=0.01, far=100.0):
    """Render scene using Gaussian Rasterization."""
    with torch.no_grad():
        cam = setup_camera(w, h, k, w2c, near, far)
        im, _, depth = Renderer(raster_settings=cam)(**timestep_data)
        return im, depth


def export_depth(scene_root, output_root, checkpoint_path):
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

    # Render and save the depths
    os.makedirs(output_root, exist_ok=True)
    rgbs = np.stack([
        np.stack([
            np.array(Image.open(frame_paths[v][t]))
            for t in range(n_frames)
        ])
        for v in range(n_views)
    ])
    h, w = rgbs.shape[2], rgbs.shape[3]
    for v, view_idx in enumerate(views):
        depths = []
        for t in range(n_frames):
            im, depth = render(w, h, k[v].numpy(), extrinsics[v].numpy(), scene_data[t])
            depths.append(depth.cpu().numpy()[0])
        depths = np.stack(depths)
        np.save(output_root / f"depths_{view_idx:02d}.npy", depths)


if __name__ == "__main__":
    print("Exporting depths from pretrained checkpoints")
    for sequence_name in tqdm(["basketball", "boxes", "football", "juggle", "softball", "tennis"]):
        scene_root = Path(f"./datasets/panoptic_d3dgs/{sequence_name}")
        output_path = Path(f"./datasets/panoptic_d3dgs/{sequence_name}/dynamic3dgs_depth")
        checkpoint_path = Path(f"./dynamic3dgs/output/pretrained/{sequence_name}")
        export_depth(scene_root, output_path, checkpoint_path)
