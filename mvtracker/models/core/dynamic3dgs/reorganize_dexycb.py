import os

source_roots = [f for f in os.listdir(".") if f.startswith("2020")]
import os
import shutil

source_roots = [f for f in os.listdir(".") if f.startswith("2020")]
for source_root in source_roots:
    target_root = source_root
    ims_target = os.path.join(target_root, "ims")
    seg_target = os.path.join(target_root, "seg")
    depths_target = os.path.join(target_root, "depths")

    for target in [ims_target, seg_target, depths_target]:
        os.makedirs(target, exist_ok=True)

    for i in range(8):  # view_00 to view_07
        view_folder = os.path.join(source_root, f"view_{i:02d}")

        ims_source = os.path.join(view_folder, "rgb")
        ims_dest = os.path.join(ims_target, str(i))
        if os.path.exists(ims_source):
            shutil.copytree(ims_source, ims_dest, dirs_exist_ok=True)

        mask_source = os.path.join(view_folder, "mask")
        seg_dest = os.path.join(seg_target, str(i))
        if os.path.exists(mask_source):
            shutil.copytree(mask_source, seg_dest, dirs_exist_ok=True)

        depth_source = os.path.join(view_folder, "depth")
        depth_dest = os.path.join(depths_target, str(i))
        if os.path.exists(depth_source):
            shutil.copytree(depth_source, depth_dest, dirs_exist_ok=True)

print("Copying complete!")
