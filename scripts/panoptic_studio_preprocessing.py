"""
This script will convert the Panoptic Studio subset of TAPVid-3D to multi-view 3D point tracking dataset.

First, follow the instructions at https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid3d
to download the raw panoptic studio data, for example, as follows:
```bash
# Set up a temporary environment
conda create -n panoptic-preprocessing python=3.10.12 -y
conda activate panoptic-preprocessing
pip install "git+https://github.com/google-deepmind/tapnet.git#egg=tapnet[tapvid3d_eval,tapvid3d_generation]"

# Download the raw data
python -m tapnet.tapvid3d.annotation_generation.generate_pstudio --output_dir datasets/panoptic_studio_tapvid3d
mkdir datasets/panoptic-multiview
mv datasets/panoptic_studio_tapvid3d/tmp/data/* datasets/panoptic-multiview/

# If you like, you can remove the temporary environment now
conda deactivate
conda env remove -n panoptic-preprocessing
```

Following https://github.com/JonathonLuiten/Dynamic3DGaussians#run-visualizer-on-pretrained-models,
download and unzip the pretrained Dynamic3DGS checkpoints, e.g. as follows:
```bash
wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/output.zip -O checkpoints/output.zip
unzip checkpoints/output.zip -d checkpoints/
rm checkpoints/output.zip
mv checkpoints/output/pretrained checkpoints/dynamic3dgs_pretrained
```

Install the missing dependencies needed by Dynamic3DGS:
```bash
conda activate 3dpt
conda install -c conda-forge gcc_linux-64=11.3.0 gxx_linux-64=11.3.0 gcc=11.3.0 gxx=11.3.0 -y
pip install git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git
pip install open3d==0.16.0
```

Now you can run this script to generate the Dynamic3DGS depths and merge the TAP-Vid3D annotations:
```bash
python -m scripts.panoptic_studio_preprocessing \
  --dataset_root ./datasets/panoptic-multiview \
  --checkpoint_root ./checkpoints/dynamic3dgs_pretrained \
  --tapvid3d_root ./datasets/panoptic_studio_tapvid3d
```

The processed dataset is now stored in ./datasets/panoptic-multiview.
If you'd like, you can remove the raw tapvid3d data now to save space:
```bash
rm -rf ./datasets/panoptic_studio_tapvid3d
```
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from mvtracker.models.core.dynamic3dgs.export_depths_from_pretrained_checkpoint import export_depth
from mvtracker.models.core.dynamic3dgs.merge_tapvid3d_per_camera_annotations import merge_annotations


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Panoptic Studio TAPVid-3D subset.")
    parser.add_argument("--dataset_root", type=Path, required=True,
                        help="Root path to Panoptic Studio dataset (per-sequence folders).")
    parser.add_argument("--checkpoint_root", type=Path, required=True,
                        help="Root path to Dynamic3DGS pretrained checkpoints (per-sequence).")
    parser.add_argument("--tapvid3d_root", type=Path, required=True,
                        help="Root path to TAPVid-3D annotations for Panoptic Studio.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sequences = ["basketball", "boxes", "football", "juggle", "softball", "tennis"]

    print("Exporting depths from pretrained checkpoints")
    for sequence_name in tqdm(sequences):
        scene_root = args.dataset_root / sequence_name
        output_path = scene_root / "dynamic3dgs_depth"
        checkpoint_path = args.checkpoint_root / sequence_name
        export_depth(scene_root, output_path, checkpoint_path)

    print("Merging TAP-Vid3D per-camera annotations.")
    for sequence_name in tqdm(sequences):
        scene_root = args.dataset_root / sequence_name
        checkpoint_path = args.checkpoint_root / sequence_name
        tapvid3d_annotation_paths = list(args.tapvid3d_root.glob(f"{sequence_name}_*.npz"))
        merge_annotations(
            scene_root,
            checkpoint_path,
            tapvid3d_annotation_paths,
            skip_if_output_already_exists=True,
            rerun_logging=True
        )
