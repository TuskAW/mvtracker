import json
import os
from collections import defaultdict

import kornia
import numpy as np
import torch

BASE_PATH = "."
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
NUM_TIMESTEPS = 24
SELECTED_CAMS = [0, 1, 2, 3]
OUTPUT_NAME = "0123_metadata"
# Filter valid sequences
sequences = [f for f in os.listdir(BASE_PATH)]

for sequence in sequences:
    sequence_path = os.path.join(BASE_PATH, sequence)
    view_folders = [
        f
        for f in os.listdir(sequence_path)
        if f.startswith("view_") and f[-1:].isdigit()
    ]

    combined_data = defaultdict(
        lambda: defaultdict(
            lambda: {
                "cam_id": 0,
                "w": 0,
                "h": 0,
                "k": [],
                "w2c": [],
                "fn": [],
                "sensor_width": 0,
                "focal_length": 0,
            }
        )
    )

    if not view_folders:
        continue

    first_valid_view = None
    for vf in view_folders:
        cam_id = int(vf[-1:])
        if not SELECTED_CAMS or cam_id in SELECTED_CAMS:
            first_valid_view = vf
            break

    if first_valid_view is None:
        continue

    example_path = os.path.join(sequence_path, first_valid_view)
    all_frames = [
        f
        for f in os.listdir(example_path)
        if f.endswith(".png") and f[:-4].isdigit() and f.startswith("rgba")
    ]
    num_timesteps = len(all_frames)

    for time_step in range(NUM_TIMESTEPS):
        for view_folder in view_folders:
            print(f"Processing {sequence}/{view_folder}, time step {time_step}")

            view_folder_path = os.path.join(sequence_path, view_folder)
            if not os.path.exists(view_folder_path):
                continue

            cam_id = int(view_folder[-1:])
            if SELECTED_CAMS != [] and cam_id not in SELECTED_CAMS:
                continue

            with open(os.path.join(view_folder_path, "metadata.json"), "r") as f:
                data = json.load(f)

            cam_data = data["camera"]
            k = cam_data["K"]

            quaternions = torch.tensor(cam_data["quaternions"])
            positions = torch.tensor(cam_data["positions"])

            rot_matrices = kornia.geometry.quaternion_to_rotation_matrix(quaternions)

            ext_inv = torch.eye(4).repeat(NUM_TIMESTEPS, 1, 1)
            ext_inv[:, :3, :3] = rot_matrices
            ext_inv[:, :3, 3] = positions

            ext = ext_inv.inverse()[:, :3, :]
            ext = np.diag([1, -1, -1]) @ ext.numpy()

            w2c = ext[0].tolist()
            w2c.append([0, 0, 0, 1])

            intrinsics = (
                    np.diag([IMAGE_WIDTH, IMAGE_HEIGHT, 1])
                    @ np.array(k)
                    @ np.diag([1, -1, -1])
            )
            frame_name = f"{cam_id}/{str(time_step).zfill(5)}.png"

            cam_info = combined_data[time_step][str(cam_id)]
            cam_info["cam_id"] = cam_id
            cam_info["w"] = IMAGE_WIDTH
            cam_info["h"] = IMAGE_HEIGHT
            cam_info["k"] = intrinsics.tolist()
            cam_info["w2c"] = w2c
            cam_info["fn"] = frame_name
            cam_info["sensor_width"] = cam_data["sensor_width"]
            cam_info["focal_length"] = cam_data["focal_length"]

    output_path = os.path.join(sequence_path, f"{OUTPUT_NAME}.json")
    with open(output_path, "w") as f:
        json.dump(dict(combined_data), f, indent=4)

    print(f"Saved metadata for {sequence}")
