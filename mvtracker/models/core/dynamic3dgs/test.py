import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm

from external import calc_psnr, calc_ssim
from helpers import setup_camera

TEST_CAMS = [0, 10, 15, 30]


def load_saved_params(seq, exp):
    """Load saved parameters for testing."""
    params_path = f"./output/{exp}/{seq}/params.npz"
    params = np.load(params_path)
    params = {k: torch.tensor(v).cuda().float() for k, v in params.items()}
    return params


def prepare_test_dataset(t, md, seq, exclude_cam_ids):
    """Prepare dataset for the given timestep, excluding specific camera IDs."""
    dataset = []
    used_cam_ids = []
    for c in range(len(md["fn"][t])):
        cam_id = md["cam_id"][t][c]
        # if cam_id in exclude_cam_ids:
        #     continue
        # ONLY USE THE SPECIFIC CAMS
        if cam_id not in TEST_CAMS:
            continue
        w, h, k, w2c = md["w"], md["h"], md["k"][t][c], md["w2c"][t][c]
        cam = setup_camera(w, h, k, w2c, near=1.0, far=100)
        fn = md["fn"][t][c]
        im_path = f"./data/{seq}/ims/{fn}"
        im = np.array(Image.open(im_path)) / 255.0
        im = torch.tensor(im).float().cuda().permute(2, 0, 1)
        dataset.append({"cam": cam, "im": im, "id": cam_id})
        used_cam_ids.append(cam_id)
    return dataset, used_cam_ids


def render_image(cam, rendervar):
    """Render an image using the given camera and render variables."""
    with torch.no_grad():
        im, _, _ = Renderer(raster_settings=cam)(**rendervar)
    return im


def test(seq, exp, exclude_cam_ids=[]):
    """Test saved parameters on a dataset and report metrics."""
    print(f"Testing sequence: {seq}, experiment: {exp}")

    # Load metadata and saved parameters
    md = json.load(open(f"./data/{seq}/test_meta.json", "r"))  # metadata
    params = load_saved_params(seq, exp)

    # Prepare output paths
    render_path = f"./output/{exp}/{seq}/renders"
    results_path = f"./output/{exp}_metrics_test.csv"
    os.makedirs(render_path, exist_ok=True)

    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("Sequence,Experiment,Timestep,Camera ID,PSNR,SSIM\n")

    num_timesteps = len(md["fn"])
    psnrs, ssims = [], []
    used_cameras = []

    for t in tqdm(range(num_timesteps), desc="Testing timesteps"):
        dataset, used_cam_ids = prepare_test_dataset(t, md, seq, exclude_cam_ids)
        used_cameras.extend(used_cam_ids)
        rendervar = {
            "means3D": params["means3D"][t],
            "colors_precomp": params["rgb_colors"][t],
            "rotations": torch.nn.functional.normalize(params["unnorm_rotations"][t]),
            "opacities": torch.sigmoid(params["logit_opacities"]),
            "scales": torch.exp(params["log_scales"]),
            "means2D": torch.zeros_like(params["means3D"][t], device="cuda"),
        }

        for camera in dataset:
            im_rendered = render_image(camera["cam"], rendervar)
            gt = camera["im"]

            # Save rendered and ground truth images
            idx = camera["id"]
            torchvision.utils.save_image(
                im_rendered, f"{render_path}/t{t:03d}_c{idx:02d}_rendered.png"
            )
            torchvision.utils.save_image(
                gt, f"{render_path}/t{t:03d}_c{idx:02d}_gt.png"
            )

            # Compute metrics
            psnr_val = calc_psnr(im_rendered, gt).mean().item()
            ssim_val = calc_ssim(im_rendered, gt).mean().item()
            psnrs.append(psnr_val)
            ssims.append(ssim_val)

            # Save metrics
            with open(results_path, "a") as f:
                f.write(f"{seq},{exp},{t},{idx},{psnr_val:.4f},{ssim_val:.4f}\n")

    print(f"Used cameras: {sorted(set(used_cameras))}")
    print(f"Average PSNR: {np.mean(psnrs):.4f}, Average SSIM: {np.mean(ssims):.4f}")


if __name__ == "__main__":
    exp_name = "testing_init_pt"
    training_cam_ids = [1, 4, 7, 11, 17, 20, 23, 26, 29]  # Cameras used during training
    # for sequence in ["basketball", "boxes", "football"]:
    for sequence in ["basketball"]:
        test(sequence, exp_name, exclude_cam_ids=training_cam_ids)
