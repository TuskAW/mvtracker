"""
Merge MP4 files of different methods into a single side-by-side comparison,
adding a small text bar for each method using Pillow + ImageClip
instead of MoviePy's TextClip (which requires ImageMagick).

Usage: python merge_comparison_mp4s.py
"""

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    clips_array,
    CompositeVideoClip
)


def create_title_image(text, width, height=50, bg_color=(255, 255, 255)):
    """
    Creates a PIL Image of size (width x height) with the given text, centered.
    Returns a NumPy array (H x W x 3).
    """
    # Create a blank RGB image
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Choose a default font. If you have a TTF file, specify it here:
    font = ImageFont.truetype("times_new_roman.ttf", size=36)
    # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=24)
    # If you don't have a TTF file handy, ImageFont.load_default() is the fallback:
    # font = ImageFont.load_default()

    text_w, text_h = draw.textsize(text, font=font)
    x = (width - text_w) // 2
    y = (height - text_h) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return np.array(img)


def merge_mp4s(mp4s_title_to_path_dict, merged_mp4_output_path, num_columns):
    """
    Merges each input MP4 (which presumably has a 'first column' or 'second column'
    that you want to extract) into a side-by-side comparison video, arranged in
    multiple rows if num_columns < number_of_videos, AND places each method's
    title bar above its own clip.

    :param mp4s_title_to_path_dict: dict of {title: path_to_video}
    :param merged_mp4_output_path: output MP4 path
    :param num_columns: number of clips to display per row
    """
    titles = list(mp4s_title_to_path_dict.keys())
    raw_clips = []

    # 1) Load each video and crop the relevant half-column
    for title in titles:
        path = mp4s_title_to_path_dict[title]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        clip = VideoFileClip(path)

        w, h = clip.size  # (width, height)
        if "GT" in title:
            # Crop the first column
            sub_clip = clip.crop(x1=0, x2=w // 2, y1=0, y2=h)
        else:
            # Crop the second column
            sub_clip = clip.crop(x1=w // 2, x2=w, y1=0, y2=h)

        raw_clips.append((title, sub_clip))

    # 2) For each sub-clip, create a small "title bar" on top
    #    so each method has its own label above its clip.
    bar_height = 50
    titled_clips = []
    for (title, subclip) in raw_clips:
        # Create a bar image for the subclip width
        title_img_array = create_title_image(title, subclip.w, bar_height)
        title_iclip = ImageClip(title_img_array, duration=subclip.duration)

        # Shift subclip downward by bar_height
        subclip_shifted = subclip.set_position((0, bar_height))

        # Composite them vertically: [title bar on top, subclip below]
        comp_h = bar_height + subclip.h
        comp_w = subclip.w
        composite = CompositeVideoClip(
            [title_iclip, subclip_shifted],
            size=(comp_w, comp_h)
        )

        titled_clips.append(composite)

    # 3) Normalize all titled_clips to the same height if they differ.
    import math
    min_height = min(tc.h for tc in titled_clips)
    normalized_clips = []
    for tc in titled_clips:
        if tc.h != min_height:
            scale = min_height / tc.h
            new_w = int(tc.w * scale)
            resized = tc.resize((new_w, min_height))
            normalized_clips.append(resized)
        else:
            normalized_clips.append(tc)

    # 4) Arrange the normalized clips in rows of length `num_columns`.
    n = len(normalized_clips)
    n_rows = math.ceil(n / num_columns)
    rows = []
    idx = 0
    for _ in range(n_rows):
        row_clips = normalized_clips[idx: idx + num_columns]
        rows.append(row_clips)
        idx += num_columns

    # 5) Stack them using clips_array
    final_clip = clips_array(rows)

    # 6) Write to output
    final_clip.write_videofile(
        merged_mp4_output_path,
        fps=12,
        codec="libx264",
        threads=4  # adjust as needed
    )
    print(f"✅ Merged video saved successfully to {merged_mp4_output_path}")


if __name__ == '__main__':
    for selection in ["A", "B", "C"]:
        if selection == "A":
            datasets_seq = [
                *[("kubric-multiview-v3-views0123-novelviews4", seq) for seq in [0, 3, 4, 5]],
                *[("panoptic-multiview-views1_7_14_20-novelviews24", seq) for seq in [0, 3, 4, 5]],
                *[("panoptic-multiview-views1_7_14_20-novelviews27", seq) for seq in [0, 3, 4, 5]],
            ]
            mp4s = {
                "GT": "logs/cotracker3-online/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "Dynamic 3DGS": "logs/dynamic_3dgs/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "Shape of Motion": "logs/shape_of_motion/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "LocoTrack": "logs/locotrack/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "CoTracker3": "logs/cotracker3-online/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "DELTA": "logs/delta/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "SpaTracker-1": "logs/kubric_v3_duster0123/multiview-adapter-pretrained-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "SpaTracker": "logs/kubric_v3/multiview-adapter-002/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_69799.mp4",
                # "SpaTracker-3": "logs/kubric_v3_duster0123/multiview-adapter-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_90799.mp4",
                "Triplane Baseline": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_99999.mp4",
                # "Triplane-2": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_99999.mp4",
                "Ours": "logs/kubric_v3_augs/mvtracker/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_159999.mp4",
            }
        elif selection == "B":
            datasets_seq = [
                *[("dex-ycb-multiview-duster0123-novelviews4", seq) for seq in [0, 3, 4, 5]],
                *[("dex-ycb-multiview-duster0123-novelviews5", seq) for seq in [0, 3, 4, 5]],
                *[("dex-ycb-multiview-duster0123-novelviews6", seq) for seq in [0, 3, 4, 5]],
                *[("dex-ycb-multiview-duster0123-novelviews7", seq) for seq in [0, 3, 4, 5]],
            ]
            mp4s = {
                "GT": "logs/cotracker3-online/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "Dynamic 3DGS": "logs/dynamic_3dgs/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "Shape of Motion": "logs/shape_of_motion/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "LocoTrack": "logs/locotrack/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "CoTracker3": "logs/cotracker3-online/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "DELTA": "logs/delta/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "SpaTracker-1": "logs/kubric_v3_duster0123/multiview-adapter-pretrained-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "SpaTracker-2": "logs/kubric_v3/multiview-adapter-002/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_69799.mp4",
                "SpaTracker": "logs/kubric_v3_duster0123/multiview-adapter-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_90799.mp4",
                # "Triplane-1": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_99999.mp4",
                "Triplane Baseline": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_99999.mp4",
                "Ours": "logs/kubric_v3_augs/mvtracker/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_159999.mp4",
            }
        elif selection == "C":
            datasets_seq = [
                *[("dex-ycb-multiview-duster2345-novelviews7", seq) for seq in [0, 3, 4, 5]],
                *[("dex-ycb-multiview-duster4567-novelviews7", seq) for seq in [0, 3, 4, 5]],
                *[("dex-ycb-multiview-duster4567-novelviews0", seq) for seq in [0, 3, 4, 5]],
            ]
            mp4s = {
                "GT": "logs/cotracker3-online/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "Dynamic 3DGS": "logs/dynamic_3dgs/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "Shape of Motion": "logs/shape_of_motion/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "LocoTrack": "logs/locotrack/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "CoTracker3": "logs/cotracker3-online/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                "DELTA": "logs/delta/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "SpaTracker-1": "logs/kubric_v3_duster0123/multiview-adapter-pretrained-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_-1.mp4",
                # "SpaTracker-2": "logs/kubric_v3/multiview-adapter-002/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_69799.mp4",
                "SpaTracker": "logs/kubric_v3_duster0123/multiview-adapter-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_90799.mp4",
                # "Triplane-1": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_99999.mp4",
                "Triplane Baseline": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_99999.mp4",
                "Ours": "logs/kubric_v3_augs/mvtracker/eval_{dataset}/comparison_v4b-novel__seq-{seq}_step_159999.mp4",
            }
        else:
            raise ValueError(f"Invalid selection: {selection}")

        for dataset, seq in datasets_seq:
            mp4s_title_to_path_dict = {
                key: path.format(dataset=dataset, seq=seq)
                for key, path in mp4s.items()
            }
            if not mp4s_title_to_path_dict:
                print(f"⚠️ Warning: No valid MP4 files found for dataset {dataset} seq {seq}. Skipping...")
                continue
            merged_mp4 = f"logs/comparison_v4__{dataset}__seq-{seq}.mp4"
            merge_mp4s(mp4s_title_to_path_dict, merged_mp4, num_columns=3)
