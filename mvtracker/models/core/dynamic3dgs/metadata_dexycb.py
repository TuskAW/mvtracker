import json
import os
from collections import defaultdict

import numpy as np

# Configurable parameters
BASE_PATH = "."
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
SELECTED_CAMS = [0, 1, 2, 3]
OUTPUT_NAME = "0123_metadata"

# Filter sequences
sequences = [f for f in os.listdir(BASE_PATH) if f.startswith("2020")]
print(sequences)

for sequence in sequences:
    sequence_path = os.path.join(BASE_PATH, sequence)
    view_folders = [
        f
        for f in os.listdir(sequence_path)
        if f.startswith("view_") and f[-2:].isdigit()
    ]

    if not view_folders:
        continue

    example_view_path = os.path.join(sequence_path, view_folders[0])
    frame_files = [
        fname
        for fname in os.listdir(example_view_path)
        if fname.endswith(".png") and fname[:-4].isdigit()
    ]
    num_timesteps = len(frame_files)
    print(f"{sequence}: Found {num_timesteps} frames in {view_folders[0]}")

    combined_data = defaultdict(
        lambda: defaultdict(
            lambda: {"cam_id": 0, "w": 0, "h": 0, "k": [], "w2c": [], "fn": []}
        )
    )

    for time_step in range(num_timesteps):
        for view_folder in view_folders:
            view_folder_path = os.path.join(sequence_path, view_folder)
            if not os.path.exists(view_folder_path):
                print(f"Skipping {view_folder_path}")
                continue

            cam_id = int(view_folder[-2:])

            if SELECTED_CAMS != [] and cam_id not in SELECTED_CAMS:
                continue

            data_path = os.path.join(view_folder_path, "intrinsics_extrinsics.npz")

            if not os.path.exists(data_path):
                print(f"Missing intrinsics_extrinsics.npz in {view_folder_path}")
                continue

            data = np.load(data_path)
            k = data["intrinsics"][:3, :3]
            w2c = data["extrinsics"][:3, :]
            w2c = np.vstack([w2c, np.array([0, 0, 0, 1])])

            frame_name = f"{cam_id}/{str(time_step).zfill(5)}.png"

            cam_info = combined_data[time_step][str(cam_id)]
            cam_info["cam_id"] = cam_id
            cam_info["w"] = IMAGE_WIDTH
            cam_info["h"] = IMAGE_HEIGHT
            cam_info["k"] = k.tolist()
            cam_info["w2c"] = w2c.tolist()
            cam_info["fn"] = frame_name

    output_path = os.path.join(sequence_path, "metadata.json")
    with open(output_path, "w") as f:
        json.dump(dict(combined_data), f, indent=4)

    print(f"Saved metadata for {sequence}")
