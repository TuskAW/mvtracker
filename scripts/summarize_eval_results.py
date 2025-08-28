import os
import re
import warnings

import pandas as pd

REMAP_KUBRIC = {
    "Method": ("", "Method"),
    "average_jaccard__dynamic": ("Dynamic Points (motion > 0.1)", "Jacc."),
    "jaccard_0.05__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.05"),
    "jaccard_0.10__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.1"),
    "jaccard_0.20__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.2"),
    "jaccard_0.40__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.4"),
    "jaccard_0.80__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.8"),
    "average_pts_within_thresh__dynamic": ("Dynamic Points (motion > 0.1)", "Loc."),
    "pts_within_0.05__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.05"),
    "pts_within_0.10__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.1"),
    "pts_within_0.20__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.2"),
    "pts_within_0.40__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.4"),
    "pts_within_0.80__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.8"),
    "survival__dynamic": ("Dynamic Points (motion > 0.1)", "Surv."),
    "occlusion_accuracy__dynamic": ("Dynamic Points (motion > 0.1)", "OA"),
    "mte_visible__dynamic": ("Dynamic Points (motion > 0.1)", "MTE"),
    "ate_visible__dynamic": ("Dynamic Points (motion > 0.1)", "ATE"),
    "fde_visible__dynamic": ("Dynamic Points (motion > 0.1)", "FDE"),
    "n__dynamic": ("Dynamic Points (motion > 0.1)", "n"),
    "v__dynamic": ("Dynamic Points (motion > 0.1)", "v"),
    "average_jaccard__very_dynamic": ("Very Dynamic", "Jacc."),
    "average_pts_within_thresh__very_dynamic": ("Very Dynamic", "Loc."),
    "survival__very_dynamic": ("Very Dynamic", "Surv."),
    "occlusion_accuracy__very_dynamic": ("Very Dynamic", "OA"),
    "mte_visible__very_dynamic": ("Very Dynamic", "MTE"),
    "average_jaccard__static": ("Static Points (motion < 0.01)", "Jacc."),
    "average_pts_within_thresh__static": ("Static Points (motion < 0.01)", "Loc."),
    "survival__static": ("Static Points (motion < 0.01)", "Surv."),
    "occlusion_accuracy__static": ("Static Points (motion < 0.01)", "OA"),
    "mte_visible__static": ("Static Points (motion < 0.01)", "MTE"),
    "average_jaccard__any": ("Any Points", "Jacc."),
    "average_pts_within_thresh__any": ("Any Points", "Loc."),
    "survival__any": ("Any Points", "Surv."),
    "occlusion_accuracy__any": ("Any Points", "OA"),
    "mte_visible__any": ("Any Points", "MTE"),
    "n_iters": ("", "#iters"),
}

REMAP_DEXYCB_V1 = {
    "Method": ("", "Method"),
    "average_jaccard__dynamic": ("Dynamic Points (motion > 0.1)", "Jacc."),
    "jaccard_0.01__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.01"),
    "jaccard_0.02__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.02"),
    "jaccard_0.05__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.05"),
    "jaccard_0.10__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.10"),
    "jaccard_0.20__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.20"),
    "average_pts_within_thresh__dynamic": ("Dynamic Points (motion > 0.1)", "Loc."),
    "pts_within_0.01__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.01"),
    "pts_within_0.02__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.02"),
    "pts_within_0.05__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.05"),
    "pts_within_0.10__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.10"),
    "pts_within_0.20__dynamic": ("Dynamic Points (motion > 0.1)", "< 0.20"),
    "survival__dynamic": ("Dynamic Points (motion > 0.1)", "Surv."),
    "occlusion_accuracy__dynamic": ("Dynamic Points (motion > 0.1)", "OA"),
    "mte_visible__dynamic": ("Dynamic Points (motion > 0.1)", "MTE"),
    "ate_visible__dynamic": ("Dynamic Points (motion > 0.1)", "ATE"),
    "fde_visible__dynamic": ("Dynamic Points (motion > 0.1)", "FDE"),
    "n__dynamic": ("Dynamic Points (motion > 0.1)", "n"),
    "v__dynamic": ("Dynamic Points (motion > 0.1)", "v"),
    "average_jaccard__very_dynamic": ("Very Dynamic", "Jacc."),
    "average_pts_within_thresh__very_dynamic": ("Very Dynamic", "Loc."),
    "survival__very_dynamic": ("Very Dynamic", "Surv."),
    "occlusion_accuracy__very_dynamic": ("Very Dynamic", "OA"),
    "average_jaccard__static": ("Static Points (motion < 0.01)", "Jacc."),
    "average_pts_within_thresh__static": ("Static Points (motion < 0.01)", "Loc."),
    "survival__static": ("Static Points (motion < 0.01)", "Surv."),
    "occlusion_accuracy__static": ("Static Points (motion < 0.01)", "OA"),
    "average_jaccard__any": ("Any Points", "Jacc."),
    "average_pts_within_thresh__any": ("Any Points", "Loc."),
    "survival__any": ("Any Points", "Surv."),
    "occlusion_accuracy__any": ("Any Points", "OA"),
    "n_iters": ("", "#iters"),
}

# Initialize remapping dictionary with the correct order
REMAP_DEXYCB_V2 = {}
REMAP_DEXYCB_V2["Method"] = ("", "Method")

# Define ordered point categories (dynamic first, then very dynamic, static, and any)
POINT_TYPES = {
    "dynamic": "Dynamic Points (motion > 0.1)",
    "very_dynamic": "Very Dynamic",
    "static": "Static Points (motion < 0.01)",
    "any": "Any Points",
    "dynamic-static-mean": "Dynamic+Static Points Mean",
}
METRICS = {
    "average_jaccard": "Jacc.",
    "jaccard": "<{threshold}",
    "average_pts_within_thresh": "Loc.",
    "pts_within": "<{threshold}",
    "survival": "Surv.",
    "occlusion_accuracy": "OA",
    "occlusion_accuracy_for_vis0": "OA(v=0)",
    "occlusion_accuracy_for_vis1": "OA(v=1)",
    "mte_visible": "MTE",
    "ate_visible": "ATE",
    "fde_visible": "FDE",
    "n": "n",
    "v": "v"
}
THRESHOLDS = ["0.01", "0.02", "0.05", "0.10", "0.20"]
for pt_key, pt_label in POINT_TYPES.items():
    for metric, metric_label in METRICS.items():
        if metric in ["jaccard", "pts_within"]:  # Threshold-based metrics
            for thresh in THRESHOLDS:
                REMAP_DEXYCB_V2[f"{metric}_{thresh}__{pt_key}"] = (pt_label, metric_label.format(threshold=thresh))
        else:  # Regular metrics
            REMAP_DEXYCB_V2[f"{metric}__{pt_key}"] = (pt_label, metric_label)
REMAP_DEXYCB_V2["n_iters"] = ("", "#iters")

REMAP_TAPVID2D_INDEX_NAMES = ["Metric Definition", "Metric"]
REMAP_TAPVID2D = {
    "Method": ("", "Method",),
    "average_jaccard__any": ("Our Metrics", "Jacc.",),
    "jaccard_1.00__any": ("Our Metrics", "<  1",),
    "jaccard_2.00__any": ("Our Metrics", "<  2",),
    "jaccard_4.00__any": ("Our Metrics", "<  4",),
    "jaccard_8.00__any": ("Our Metrics", "<  8",),
    "jaccard_16.00__any": ("Our Metrics", "< 16",),
    "average_pts_within_thresh__any": ("Our Metrics", "Loc.",),
    "pts_within_1.00__any": ("Our Metrics", "<  1",),
    "pts_within_2.00__any": ("Our Metrics", "<  2",),
    "pts_within_4.00__any": ("Our Metrics", "<  4",),
    "pts_within_8.00__any": ("Our Metrics", "<  8",),
    "pts_within_16.00__any": ("Our Metrics", "< 16",),
    "survival__any": ("Our Metrics", "Surv.",),
    "occlusion_accuracy__any": ("Our Metrics", "OA",),
    "occlusion_accuracy_for_vis0__any": ("Our Metrics", "OA(v=0)",),
    "occlusion_accuracy_for_vis1__any": ("Our Metrics", "OA(v=1)",),
    "mte_visible__any": ("Our Metrics", "MTE",),
    "ate_visible__any": ("Our Metrics", "ATE",),
    "fde_visible__any": ("Our Metrics", "FDE",),
    "n__any": ("Our Metrics", "n",),
    "v__any": ("Our Metrics", "v",),
    "tapvid2d_average_jaccard": ("TAPVid-2D Metrics", "Jacc.",),
    "tapvid2d_jaccard_1": ("TAPVid-2D Metrics", "<  1",),
    "tapvid2d_jaccard_2": ("TAPVid-2D Metrics", "<  2",),
    "tapvid2d_jaccard_4": ("TAPVid-2D Metrics", "<  4",),
    "tapvid2d_jaccard_8": ("TAPVid-2D Metrics", "<  8",),
    "tapvid2d_jaccard_16": ("TAPVid-2D Metrics", "< 16",),
    "tapvid2d_average_pts_within_thresh": ("TAPVid-2D Metrics", "Loc.",),
    "tapvid2d_pts_within_1": ("TAPVid-2D Metrics", "<  1",),
    "tapvid2d_pts_within_2": ("TAPVid-2D Metrics", "<  2",),
    "tapvid2d_pts_within_4": ("TAPVid-2D Metrics", "<  4",),
    "tapvid2d_pts_within_8": ("TAPVid-2D Metrics", "<  8",),
    "tapvid2d_pts_within_16": ("TAPVid-2D Metrics", "< 16",),
    "tapvid2d_occlusion_accuracy": ("TAPVid-2D Metrics", "OA",),
    "n_iters": ("", "#iters",),
}

REMAP_PANOPTIC = {}
REMAP_PANOPTIC["Method"] = ("", "Method")
for pt_key in ["any"]:
    pt_label = POINT_TYPES[pt_key]
    for metric, metric_label in METRICS.items():
        if metric in ["jaccard", "pts_within"]:  # Threshold-based metrics
            for thresh in ["0.05", "0.10", "0.20", "0.40"]:
                REMAP_PANOPTIC[f"{metric}_{thresh}__{pt_key}"] = (pt_label, metric_label.format(threshold=thresh))
        else:  # Regular metrics
            REMAP_PANOPTIC[f"{metric}__{pt_key}"] = (pt_label, metric_label)
REMAP_PANOPTIC["n_iters"] = ("", "#iters")

PARTIAL_REMAP_FOR_2DPT_ABLATION = {}
for pt_key, pt_label in POINT_TYPES.items():
    for metric, metric_label in METRICS.items():
        if "jaccard" in metric or "occlusion" in metric:
            continue
        if metric in ["jaccard", "pts_within"]:  # Threshold-based metrics
            for thresh in ["1.00", "2.00", "4.00", "8.00", "16.00"]:
                PARTIAL_REMAP_FOR_2DPT_ABLATION[f"2dpt__{metric}_{thresh}__{pt_key}"] = (
                    "(2DPT) " + pt_label, metric_label.format(threshold=thresh)
                )
        else:  # Regular metrics
            PARTIAL_REMAP_FOR_2DPT_ABLATION[f"2dpt__{metric}__{pt_key}"] = ("(2DPT) " + pt_label, metric_label)
for logged_key, (pt_label, metric_label) in REMAP_TAPVID2D.items():
    if "jaccard" in logged_key or "occlusion" in logged_key:
        continue
    if "tapvid2d" not in logged_key:
        continue
    PARTIAL_REMAP_FOR_2DPT_ABLATION[f"2dpt__{logged_key}"] = ("(2DPT) " + pt_label, metric_label)
REMAP_2DPT_ABLATION = REMAP_KUBRIC | PARTIAL_REMAP_FOR_2DPT_ABLATION

ONE_REMAP_TO_RULE_THEM_ALL = {}
ONE_REMAP_TO_RULE_THEM_ALL["Method"] = ("", "Method")
ONE_REMAP_TO_RULE_THEM_ALL["Dataset"] = ("", "Dataset")
THRESHOLDS = ["0.01", "0.02", "0.05", "0.10", "0.20", "0.40"]
for pt_key, pt_label in POINT_TYPES.items():
    for metric, metric_label in METRICS.items():
        if metric in ["jaccard", "pts_within"]:  # Threshold-based metrics
            for thresh in THRESHOLDS:
                ONE_REMAP_TO_RULE_THEM_ALL[f"{metric}_{thresh}__{pt_key}"] = (
                    pt_label, metric_label.format(threshold=thresh))
        else:  # Regular metrics
            ONE_REMAP_TO_RULE_THEM_ALL[f"{metric}__{pt_key}"] = (pt_label, metric_label)
ONE_REMAP_TO_RULE_THEM_ALL["n_iters"] = ("", "#iters")


def find_file_with_max_steps(folder):
    if not os.path.isdir(folder):
        return None, -1
    pattern = re.compile(r"step-(\d+)_metrics_avg.csv")
    max_steps = -1
    max_file = None
    for filename in os.listdir(folder):
        m = pattern.search(filename)
        if m:
            steps = int(m.group(1))
            if steps > max_steps:
                max_steps = steps
                max_file = filename
    return max_file, max_steps


def create_table(
        method_name_to_csv_path,
        remap=REMAP_KUBRIC,
        remap_index_names=["Type", "Metric"],
        header=True,
        skip_missing=False,
):
    assert len(method_name_to_csv_path) > 0, "No CSV files provided"
    rows = []
    order = []
    for method_name, path in method_name_to_csv_path.items():
        if "step-?_" in path:
            filename, n_iters = find_file_with_max_steps(os.path.dirname(path))
            if filename is None:
                warnings.warn(f"No CSV files found in {os.path.dirname(path)}")
                continue
            path = os.path.join(os.path.dirname(path), filename)
        if not os.path.exists(path):
            if skip_missing:
                warnings.warn(f"Skipping missing file: {path}")
                continue
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path, header=None, names=["Metric", "Value"])
        df = df.dropna(subset=["Metric"]).reset_index(drop=True)
        if type(method_name) == tuple:
            method_name, dataset_name = method_name
        else:
            dataset_name = os.path.basename(os.path.dirname(path)).replace("eval_", "")
        df["Method"] = method_name
        match = re.search(r"step-(\d+)", path)
        n_iters = int(match.group(1)) if match else 0
        df.loc[len(df)] = ["n_iters", n_iters, method_name]
        df["Metric"] = df["Metric"].str.split("/").str[-1].str.replace("model__", "")
        df["Dataset"] = dataset_name
        rows.append(df)
        order.append((method_name, dataset_name))
    combined_df = pd.concat(rows)
    pivot_df = combined_df.pivot(index=["Method", "Dataset"], columns="Metric", values="Value").reset_index()

    pivot_df = pivot_df.set_index(["Method", "Dataset"]).reindex(order).reset_index()

    # Define a mapping for the new names
    for k in remap.keys():
        if k not in pivot_df.columns:
            pivot_df[k] = None
            pivot_df = pivot_df.copy()  # To avoid "DataFrame is highly fragmented" warning
    pivot_df = pivot_df[remap.keys()]
    multi_index = pd.MultiIndex.from_tuples(
        tuples=[remap[col] for col in pivot_df.columns],
        names=remap_index_names,
    )
    pivot_df.columns = multi_index

    return pivot_df, pivot_df.to_csv(index=False, header=header)


def kubric_single_point():
    print("Kubric single-point evaluation results:")
    print("================================")
    df, csv_str = create_table({
        # ls logs/kubric_v3/*/eval_kubric-multiview-v3-single/step-*_kubric-multiview-v3-single_metrics_avg.csv | cat
        "SpaTracker (pretrained)": "logs/kubric_v3/multiview-adapter-pretrained-004/eval_kubric-multiview-v3-single/step--1_kubric-multiview-v3-single_metrics_avg.csv",
        "SpaTracker (single-view baseline)": "logs/kubric_v3/multiview-adapter-002/eval_kubric-multiview-v3-single/step-69799_kubric-multiview-v3-single_metrics_avg.csv",
        "Multi-view-V1 (ours)": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3-single/step-99999_kubric-multiview-v3-single_metrics_avg.csv",
        "Multi-view-V2 (ours)": "logs/kubric_v3/multiview-v2-002--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-single/step-91599_kubric-multiview-v3-single_metrics_avg.csv",
    })
    print(csv_str)


def kubric_before_gt0123():
    print("Kubric multi-point evaluation results:")
    print("================================")
    df, csv_str = create_table({
        # ls logs/kubric_v3/*/eval_kubric-multiview-v3/step-*_kubric-multiview-v3_metrics_avg.csv | cat
        "CopyCat (No motion baseline)": "logs/copycat/eval_kubric-multiview-v3/step--1_kubric-multiview-v3_metrics_avg.csv",
        "SpaTracker (pretrained)": "logs/kubric_v3/multiview-adapter-pretrained-004/eval_kubric-multiview-v3/step--1_kubric-multiview-v3_metrics_avg.csv",
        "SpaTracker (single-view baseline)": "logs/kubric_v3/multiview-adapter-002/eval_kubric-multiview-v3/step-69799_kubric-multiview-v3_metrics_avg.csv",
        "Multi-view-V1 (ours)": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3/step-99999_kubric-multiview-v3_metrics_avg.csv",
        "Multi-view-V2 (ours)": "logs/kubric_v3/multiview-v2-002--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3/step-91599_kubric-multiview-v3_metrics_avg.csv",

        "Multi-view-V1 (ours) (128; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3/step-100000_kubric-multiview-v3_metrics_avg.csv",
        "Multi-view-V1 (ours) (256; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_kubric-multiview-v3/step-100000_kubric-multiview-v3_metrics_avg.csv",
        "Multi-view-V2 (ours) (finetuned on D4c)": "logs/kubric_v3_duster0123/multiview-v2-pretrained-cleaned-003--lr-2.5e-4--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3/step-10000_kubric-multiview-v3_metrics_avg.csv",
        "Multi-view-V2 (ours) (trained on D4)": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3/step-99999_kubric-multiview-v3_metrics_avg.csv",
    })
    print(csv_str)


def kubric():
    print("Kubric multi-point evaluation results:")
    print("================================")
    df, csv_str = create_table({
        "CopyCat (No motion baseline)": "logs/copycat/eval_kubric-multiview-v3-gt0123/step--1_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "SpaTracker (pretrained)": "logs/kubric_v3/multiview-adapter-pretrained-004/eval_kubric-multiview-v3-gt0123/step--1_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "SpaTracker (single-view baseline)": "logs/kubric_v3/multiview-adapter-002/eval_kubric-multiview-v3-gt0123/step-69799_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "Multi-view-V1 (ours)": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3-gt0123/step-99999_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "Multi-view-V2 (ours)": "logs/kubric_v3/multiview-v2-002--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-gt0123/step-91599_kubric-multiview-v3-gt0123_metrics_avg.csv",

        "Multi-view-V1 (ours) (128; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3-gt0123/step-99999_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "Multi-view-V1 (ours) (256; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_kubric-multiview-v3-gt0123/step-99999_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "Multi-view-V2 (ours) (finetuned on D4c)": "logs/kubric_v3_duster0123/multiview-v2-pretrained-cleaned-003--lr-2.5e-4--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-gt0123/step-9999_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "Multi-view-V2 (ours) (trained on D4)": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-gt0123/step-99999_kubric-multiview-v3-gt0123_metrics_avg.csv",
        "Multi-view-V2 (ours) (trained on D4c)": "logs/kubric_v3_duster0123/multiview-v2-cleaned-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-gt0123/step-99999_metrics_avg.csv",
        "Multi-view-V3 (ours) (finetuned^2 on D4c;s=4)": "logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-gt0123/step-9999_kubric-multiview-v3-gt0123_metrics_avg.csv",
    })
    print(csv_str)


def kubric_duster():
    print("Kubric multi-point evaluation results, Duster0123:")
    print("================================")
    df, csv_str = create_table({
        # ls logs/kubric_v3/*/eval_kubric-multiview-v3-duster0123/step-*_kubric-multiview-v3-duster0123_metrics_avg.csv | cat
        "CopyCat (No motion baseline)": "logs/copycat/eval_kubric-multiview-v3-duster0123/step--1_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "SpaTracker (pretrained)": "logs/kubric_v3/multiview-adapter-pretrained-004/eval_kubric-multiview-v3-duster0123/step--1_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "SpaTracker (single-view baseline)": "logs/kubric_v3/multiview-adapter-002/eval_kubric-multiview-v3-duster0123/step-69799_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "Multi-view-V1 (ours)": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3-duster0123/step-99999_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "Multi-view-V2 (ours)": "logs/kubric_v3/multiview-v2-002--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123/step-91599_kubric-multiview-v3-duster0123_metrics_avg.csv",

        "SpaTracker (single-view baseline) (trained on D4)": "logs/kubric_v3_duster0123/multiview-adapter-001/eval_kubric-multiview-v3-duster0123/step-90000_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "Multi-view-V1 (ours) (128; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3-duster0123/step-100000_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "Multi-view-V1 (ours) (256; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_kubric-multiview-v3-duster0123/step-100000_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "Multi-view-V2 (ours) (finetuned on D4)": "logs/kubric_v3_duster0123/multiview-v2-pretrained-cleaned-003--lr-2.5e-4--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123/step-10000_kubric-multiview-v3-duster0123_metrics_avg.csv",
        "Multi-view-V2 (ours) (trained on D4)": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123/step-100000_kubric-multiview-v3-duster0123_metrics_avg.csv",
        # "Multi-view-V2 (ours) (trained on D4c)": "logs/kubric_v3_duster0123/multiview-v2-cleaned-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123/step-70000_kubric-multiview-v3-duster0123_metrics_avg.csv",
        # "Multi-view-V3 (ours) (finetuned^2 on D4c;s=4)": "logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123/step-10000_kubric-multiview-v3-duster0123_metrics_avg.csv",
    })
    print(csv_str)
    df, csv_str = create_table({
        # "SpaTracker (single-view baseline) (trained on D4)": "logs/kubric_v3_duster0123/multiview-adapter-001/eval_kubric-multiview-v3-duster0123cleaned/step-90000_kubric-multiview-v3-duster0123cleaned_metrics_avg.csv",
        # "Multi-view-V1 (ours) (128; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-128-triplane-001/eval_kubric-multiview-v3-duster0123cleaned/step-100000_kubric-multiview-v3-duster0123cleaned_metrics_avg.csv",
        # "Multi-view-V1 (ours) (256; trained on D4)": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_kubric-multiview-v3-duster0123cleaned/step-100000_kubric-multiview-v3-duster0123cleaned_metrics_avg.csv",
        # "Multi-view-V2 (ours) (finetuned on D4)": "logs/kubric_v3_duster0123/multiview-v2-pretrained-cleaned-003--lr-2.5e-4--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123cleaned/step-10000_kubric-multiview-v3-duster0123cleaned_metrics_avg.csv",
        # "Multi-view-V2 (ours) (trained on D4)": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123cleaned/step-100000_kubric-multiview-v3-duster0123cleaned_metrics_avg.csv",
        "Multi-view-V2 (ours) (trained on D4c)": "logs/kubric_v3_duster0123/multiview-v2-cleaned-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123cleaned/step-70000_kubric-multiview-v3-duster0123cleaned_metrics_avg.csv",
        "Multi-view-V3 (ours) (finetuned^2 on D4c;s=4)": "logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-10000_kubric-multiview-v3-duster0123cleaned_metrics_avg.csv",
        # "Multi-view-V3 (ours) (trained on D4c;s=4)": "TBD",
        # "Multi-view-V3 (ours) (trained on D4c;s=16)": "TBD",
    })


def mv3_kubric_duster_transformed():
    print("Kubric transformed, Duster0123, Multi-view-V3 (ours) (finetuned^2 on D4c;s=4):")
    print("================================")
    df, csv_str = create_table({
        "Kubric (no world space transformations)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.00_no_transform.csv",
        "Kubric (translated by z-10)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.01_translate_z-10.csv",
        "Kubric (translated by x+4, y+4)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.02a_translate_x+4_y+4.csv",
        "Kubric (translated by x+10, y+10)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.02b_translate_x+10_y+10.csv",
        "Kubric (rotated x+90)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.03_rotate_x+90.csv",
        "Kubric (rotated y+90)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.04_rotate_y+90.csv",
        "Kubric (rotated z+90)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.05_rotate_z+90.csv",
        "Kubric (scaled down 2x)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.06_scale_down_2x.csv",
        "Kubric (scaled down 8x)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.07_scale_down_8x.csv",
        "Kubric (scaled up 2x)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.08_scale_up_2x.csv",
        "Kubric (scaled up 8x)": f"logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_kubric-multiview-v3-duster0123cleaned/step-9999_kubric-multiview-v3-duster0123cleaned_metrics_avg.09_scale_up_8x.csv",
    })
    ""
    print(csv_str)


def mv3_kubric_nviews():
    print("eval_kubric-multiview-v3-views..., Multi-view-V2 (ours) (trained on D4):")
    print("================================")
    df, csv_str = create_table({
        "1": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views0/step-99999_metrics_avg.csv",
        "2": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views01/step-99999_metrics_avg.csv",
        "3": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views012/step-99999_metrics_avg.csv",
        "4": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views0123/step-99999_metrics_avg.csv",
        "5": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views01234/step-99999_metrics_avg.csv",
        "6": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views012345/step-99999_metrics_avg.csv",
        "7": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views0123456/step-99999_metrics_avg.csv",
        "8": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views01234567/step-99999_metrics_avg.csv",
        "9": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views012345678/step-99999_metrics_avg.csv",
        "10": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-views0123456789/step-99999_metrics_avg.csv",
    })
    print(csv_str)


def mv3_kubric_duster_nviews():
    print("eval_kubric-multiview-v3-duster0123-views..., Multi-view-V2 (ours) (trained on D4):")
    print("================================")
    df, csv_str = create_table({
        "1": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123-views0/step-99999_metrics_avg.csv",
        "2": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123-views01/step-99999_metrics_avg.csv",
        "3": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123-views012/step-99999_metrics_avg.csv",
        "4": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster0123-views0123/step-99999_metrics_avg.csv",
        "5": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster01234567-views01234/step-99999_metrics_avg.csv",
        "6": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster01234567-views012345/step-99999_metrics_avg.csv",
        "7": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster01234567-views0123456/step-99999_metrics_avg.csv",
        "8": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_kubric-multiview-v3-duster01234567-views01234567/step-99999_metrics_avg.csv",
    })
    print(csv_str)


def kubric_nviews():
    print("=" * 80)
    print("=" * 80)
    print("=" * 80)
    method_name_to_csv_path_template = {
        "CopyCat (No motion baseline),{}": "logs/copycat/eval_{}/step--1_metrics_avg.csv",
        "SpaTracker (pretrained),{}": "logs/kubric_v3/multiview-adapter-pretrained-004/eval_{}/step--1_metrics_avg.csv",
        "SpaTracker (single-view baseline),{}": "logs/kubric_v3/multiview-adapter-002/eval_{}/step-69799_metrics_avg.csv",
        "Multi-view-V1 (ours),{}": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_{}/step-99999_metrics_avg.csv",
        # "Multi-view-V2 (ours),{}": "logs/kubric_v3/multiview-v2-002--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_{}/step-91599_metrics_avg.csv",

        "SpaTracker (single-view baseline) (trained on D4),{}": "logs/kubric_v3_duster0123/multiview-adapter-001/eval_{}/step-logs/kubric_v3_duster0123/multiview-adapter-001_metrics_avg.csv",
        # "Multi-view-V1 (ours) (128; trained on D4),{}": "logs/kubric_v3_duster0123/multiview-v1-with-128-triplane-001/eval_{}/step-99999_metrics_avg.csv",
        "Multi-view-V1 (ours) (256; trained on D4),{}": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_{}/step-99999_metrics_avg.csv",
        # "Multi-view-V2 (ours) (finetuned on D4c),{}": "logs/kubric_v3_duster0123/multiview-v2-pretrained-cleaned-003--lr-2.5e-4--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_{}/step-9999_metrics_avg.csv",
        "Multi-view-V2 (ours) (trained on D4),{}": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_{}/step-99999_metrics_avg.csv",
        # "Multi-view-V2 (ours) (trained on D4c),{}": "logs/kubric_v3_duster0123/multiview-v2-cleaned-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_{}/step-99999_metrics_avg.csv",
        # "Multi-view-V3 (ours) (finetuned^2 on D4c;s=4),{}": "logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_{}/step-9999_metrics_avg.csv",
    }
    method_name_to_csv_path_per_dataset = {}
    for dataset_prefix in [
        "kubric-multiview-v3-views",
        "kubric-multiview-v3-duster0123-views",
        "kubric-multiview-v3-duster01234567-views",
        "kubric-multiview-v3-duster0123cleaned-views",
        "kubric-multiview-v3-duster01234567cleaned-views",
    ]:
        method_name_to_csv_path_per_dataset[dataset_prefix] = {}
        for method_name_template, csv_path_template in method_name_to_csv_path_template.items():
            for n in range(8):
                if ("-duster0123-" in dataset_prefix or "-duster0123cleaned-" in dataset_prefix) and n > 4:
                    continue
                if ("-duster01234567-" in dataset_prefix or "-duster01234567cleaned-" in dataset_prefix) and n < 5:
                    continue
                dataset = dataset_prefix + "".join(str(i) for i in range(n + 1))
                method_name = method_name_template.format(n + 1)
                csv_path = csv_path_template.format(dataset)
                assert method_name not in method_name_to_csv_path_per_dataset[
                    dataset_prefix], f"Duplicate method name: {method_name}"
                method_name_to_csv_path_per_dataset[dataset_prefix][method_name] = csv_path
    for dataset_prefix, method_name_to_csv_path in method_name_to_csv_path_per_dataset.items():
        print(method_name_to_csv_path)
        print(f"Kubric multi-point evaluation results, {dataset_prefix}:")
        print("================================")
        df, csv_str = create_table(method_name_to_csv_path)
        print(csv_str)


MODELS = {
    "copycat": {
        "name": "CopyCat (No motion baseline)",
        "csv": "logs/copycat/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "cotracker3": {
        "name": "CoTracker3 Offline (x)",
        "csv": "logs/cotracker3/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "cotracker3offline": {
        "name": "CoTracker3 Offline",
        "csv": "logs/cotracker3-offline/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "cotracker3online": {
        "name": "CoTracker3 Online",
        "csv": "logs/cotracker3-online/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "cotracker2offline": {
        "name": "CoTracker2 Offline",
        "csv": "logs/cotracker2-offline/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "cotracker2online": {
        "name": "CoTracker2 Online",
        "csv": "logs/cotracker2-online/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "cotracker1offline": {
        "name": "CoTracker1 Offline",
        "csv": "logs/cotracker1-offline/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "cotracker1online": {
        "name": "CoTracker1 Online",
        "csv": "logs/cotracker1-online/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "delta": {
        "name": "DELTA",
        "csv": "logs/delta/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "locotrack": {
        "name": "LocoTrack",
        "csv": "logs/locotrack/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "scenetracker": {
        "name": "SceneTracker",
        "csv": "logs/scenetracker/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "spatracker-pretrained": {
        "name": "SpaTracker (pretrained)",
        "csv": "logs/kubric_v3_duster0123/multiview-adapter-pretrained-001/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "spatracker": {
        "name": "SpaTracker (single-view baseline)",
        "csv": "logs/kubric_v3/multiview-adapter-002/eval_{dataset}/step-69799_metrics_avg.csv",
    },
    "mv1": {
        "name": "Multi-view-V1 (ours)",
        "csv": "logs/kubric_v3/multiview-v1-with-128-triplane-001/eval_{dataset}/step-99999_metrics_avg.csv",
    },
    "mv2": {
        "name": "Multi-view-V2 (ours)",
        "csv": "logs/kubric_v3/multiview-v2-002--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_{dataset}/step-91599_metrics_avg.csv",
    },
    "spatracker-d4": {
        "name": "SpaTracker (single-view baseline) (trained on D4)",
        "csv": "logs/kubric_v3_duster0123/multiview-adapter-001/eval_{dataset}/step-90799_metrics_avg.csv",
    },
    "mv1-d4": {
        "name": "Multi-view-V1 (ours) (256; trained on D4)",
        "csv": "logs/kubric_v3_duster0123/multiview-v1-with-256-triplane-001/eval_{dataset}/step-99999_metrics_avg.csv",
    },
    "mv2-d4": {
        "name": "Multi-view-V2 (ours) (trained on D4)",
        "csv": "logs/kubric_v3_duster0123/multiview-v2-001--k-16--fmaps-384--groups-1--levels-4--grad-clip-1--iters-4--window-12/eval_{dataset}/step-99999_metrics_avg.csv",
    },
    "mv3-d4c": {
        "name": "Multi-view-V3 (ours) (finetuned^2 on D4c;s=4)",
        "csv": "logs/kubric_v3_duster0123/multiview-v3-001--lr-2.5e-4--fmaps-384/eval_{dataset}/step-9999_metrics_avg.csv",
    },
    "mv4-a07": {
        "name": "Multi-view-V4 (ours) (A07)",
        "csv": "logs/kubric_v3_augs/multiview-v4-A07.augs_4.002/eval_{dataset}/step-25599_metrics_avg.csv",
    },
    "mv4-b01": {
        "name": "Multi-view-V4 (ours) (B01)",
        "csv": "logs/kubric_v3_augs/multiview-v4-B01.vary_n_views.004/eval_{dataset}/step-199999_metrics_avg.csv",
    },
    "mv4-b02": {
        "name": "Multi-view-V4 (ours) (B02)",
        "csv": "logs/kubric_v3_augs/multiview-v4-B02.vary_depth_type.002a/eval_{dataset}/step-199999_metrics_avg.csv",
    },
    "mv4-b03": {
        "name": "Multi-view-V4 (ours) (B03)",
        "csv": "logs/kubric_v3_augs/multiview-v4-B03.vary_both.004/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "mv4-b03-paper": {
        "name": "Multi-view-V4 (ours) (B03 paper ckpt)",
        "csv": "logs/kubric_v3_augs/multiview-v4-B03.vary_both.004/eval_{dataset}/step-153999_metrics_avg.csv",
    },

    # # "C01.001.0" : {
    # #     "name": "Ablation (C01) – Offset 1 AddXYZ 0 K 16 P 4",
    # #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.0_K-16_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # # },
    # "C01.001.1" : {
    #     "name": "Ablation (C01) – Offset 0 AddXYZ 0",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.1_K-16_FMAP-128_PYR-4_KNN-remove_offset/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    # "C01.001.2" : {
    #     "name": "Ablation (C01) – Offset 1 AddXYZ 1",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.2_K-16_FMAP-128_PYR-4_KNN-add_neighbor_xyz/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    # # "C01.001.3" : {
    # #     "name": "Ablation (C01) – Offset 0 AddXYZ 1",
    # #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.3_K-16_FMAP-128_PYR-4_KNN-remove_offset_and_add_neighbor_xyz/eval_{dataset}/step-?_metrics_avg.csv",
    # # },
    # "C01.001.4" : {
    #     "name": "Ablation (C01) – K 1",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.4_K-1_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    # "C01.001.5" : {
    #     "name": "Ablation (C01) – K 4",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.5_K-4_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    # "C01.001.6" : {
    #     "name": "Ablation (C01) – K 8",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.6_K-8_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    # # "C01.001.7" : {
    # #     "name": "Ablation (C01) – K 32",
    # #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.7_K-32_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # # },
    # # "C01.001.8" : {
    # #     "name": "Ablation (C01) – K 64",
    # #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.8_K-64_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # # },
    # "C01.001.9" : {
    #     "name": "Ablation (C01) – P 1",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.9_K-16_FMAP-128_PYR-1_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    # "C01.001.10" : {
    #     "name": "Ablation (C01) – P 2",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.10_K-16_FMAP-128_PYR-2_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    # # "C01.001.11" : {
    # #     "name": "Ablation (C01) – P 6",
    # #     "csv": "logs/kubric_v3_augs/ablate-correlation.001.11_K-16_FMAP-128_PYR-6_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # # },

    "C02.001.0": {
        "name": "Ablation (C02) – Offset 1 AddXYZ 0 K 16 P 4",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.0_K-16_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "C02.001.1": {

        "name": "Ablation (C02) – Offset 0 AddXYZ 0",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.1_K-16_FMAP-128_PYR-4_KNN-remove_offset/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "C02.001.2": {
        "name": "Ablation (C02) – Offset 1 AddXYZ 1",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.2_K-16_FMAP-128_PYR-4_KNN-add_neighbor_xyz/eval_{dataset}/step-?_metrics_avg.csv",
    },
    # "C02.001.3" : {
    #     "name": "Ablation (C02) – Offset 0 AddXYZ 1",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.3_K-16_FMAP-128_PYR-4_KNN-remove_offset_and_add_neighbor_xyz/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    "C02.001.4": {
        "name": "Ablation (C02) – K 1",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.4_K-1_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "C02.001.5": {
        "name": "Ablation (C02) – K 4",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.5_K-4_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "C02.001.6": {
        "name": "Ablation (C02) – K 8",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.6_K-8_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    # "C02.002.7" : {
    #     "name": "Ablation (C02) – K 32",
    #     "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.002.7_K-32_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    # },
    "C02.001.8": {
        "name": "Ablation (C02) – K 64",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.8_K-64_FMAP-128_PYR-4_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "C02.001.9": {
        "name": "Ablation (C02) – P 1",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.9_K-16_FMAP-128_PYR-1_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "C02.001.10": {
        "name": "Ablation (C02) – P 2",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.10_K-16_FMAP-128_PYR-2_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "C02.001.11": {
        "name": "Ablation (C02) – P 6",
        "csv": "logs/kubric_v3_augs/ablate-correlation.dusterdepths.C02.001.11_K-16_FMAP-128_PYR-6_KNN-default/eval_{dataset}/step-?_metrics_avg.csv",
    },
    "shape-of-motion": {
        "name": "Shape of Motion (MV)",
        "csv": "logs/shape_of_motion/eval_{dataset}/step--1_metrics_avg.csv",
    },

    # June 2025
    "mvtracker-march": {
        "name": "MV-Tracker (ours; March 2025)",
        "csv": "logs/eval/mvtracker-iccv-march2025/eval_{dataset}/step--1_metrics_avg.csv",
    },
    "mvtracker-june": {
        "name": "MV-Tracker (ours; June 2025)",
        "csv": "logs/eval/mvtracker-june2025/eval_{dataset}/step--1_metrics_avg.csv",
    },
}


def tavid2d_davis():
    print("TAPVid-2D DAVIS:")
    print("================")
    models_to_report = [
        "copycat",
        "locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
        "cotracker1offline", "cotracker2offline", "cotracker3offline",
        "spatracker-pretrained",
        "spatracker", "spatracker-d4", "mv1-d4", "mv2-d4",
        "mv4-b01", "mv4-b02", "mv4-b03",
    ]
    assert all(m in MODELS for m in models_to_report)
    for resolution in [
        "-256x256",
        # "",
    ]:
        for depth_estimator in [
            # "zoedepth",
            # "moge",
            "mogewithextrinsics",
        ]:
            df, csv_str = create_table({
                MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=f"tapvid2d-davis-{depth_estimator}{resolution}")
                for m in models_to_report
            }, remap=REMAP_TAPVID2D, remap_index_names=REMAP_TAPVID2D_INDEX_NAMES)
            print(f"Resolution: {resolution}, Depth estimator: {depth_estimator}")
            print(csv_str)
            print()


def dexycb():
    print("DexYCB evaluation results:")
    print("==========================")
    for models_to_report, depths in [
        (["copycat",
          "locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
          "cotracker1offline", "cotracker2offline", "cotracker3offline",
          "spatracker-pretrained",
          "spatracker", "mv1", "mv2",
          "spatracker-d4", "mv1-d4", "mv2-d4", "mv3-d4c",
          "mv4-b01", "mv4-b02", "mv4-b03"], ""),
        (["copycat",
          "locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
          "cotracker1offline", "cotracker2offline", "cotracker3offline",
          "spatracker-pretrained",
          "spatracker", "mv1", "mv2",
          "spatracker-d4", "mv1-d4", "mv2-d4",
          "mv4-b01", "mv4-b02", "mv4-b03", "shape-of-motion", "mv4-b03-paper"], "-duster0123"),
        (["locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
          "cotracker1offline", "cotracker2offline", "cotracker3offline",
          "mv3-d4c",
          "mv4-b01", "mv4-b02", "mv4-b03"], "-duster0123cleaned"),
    ]:
        assert all(m in MODELS for m in models_to_report)
        # for remove_hand in ["", "-removehand"]:
        for remove_hand in [""]:
            df, csv_str = create_table({
                MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=f"dex-ycb-multiview{depths}{remove_hand}")
                for m in models_to_report
            }, remap=REMAP_DEXYCB_V2)
            print(f"Depths: {depths} Remove hand: {remove_hand}")
            print(csv_str)
            print()


def kubric_refactored():
    print("Kubric evaluation results:")
    print("==========================")
    for models_to_report, depths in [
        (["copycat",
          "locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
          "cotracker1offline", "cotracker2offline", "cotracker3offline",
          "spatracker-pretrained",
          "spatracker", "mv1", "mv2",
          "spatracker-d4", "mv1-d4", "mv2-d4",
          "mv4-b01", "mv4-b02", "mv4-b03", "shape-of-motion", "mv4-b03-paper"], "-views0123"),
        (["copycat",
          "locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
          "cotracker1offline", "cotracker2offline", "cotracker3offline",
          # "spatracker-pretrained",
          # "spatracker",
          "mv1", "mv2",
          # "spatracker-d4",
          "mv1-d4", "mv2-d4",
          "mv4-b01", "mv4-b02", "mv4-b03"], "-duster0123"),
        (["spatracker", "spatracker-d4", ], "-duster0123-views0123"),
        (["copycat",
          "locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
          "cotracker1offline", "cotracker2offline", "cotracker3offline",
          # "spatracker-pretrained",
          # "spatracker",
          "mv1", "mv2",
          # "spatracker-d4",
          "mv1-d4", "mv2-d4",
          "mv4-b01", "mv4-b02", "mv4-b03"], "-duster0123cleaned"),
        (["spatracker-d4"], "-duster0123cleaned-views0123"),
    ]:
        assert all(m in MODELS for m in models_to_report)
        df, csv_str = create_table({
            MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=f"kubric-multiview-v3{depths}")
            for m in models_to_report
        }, remap=REMAP_KUBRIC)
        print(f"Depths: {depths}")
        print(csv_str)
        print()


def panoptic():
    print("Panoptic Studio evaluation results:")
    print("===================================")
    models_to_report = [
        "copycat",
        "locotrack", "scenetracker", "delta", "cotracker1online", "cotracker2online", "cotracker3online",
        "cotracker1offline", "cotracker2offline", "cotracker3offline",
        "spatracker-pretrained",
        "spatracker", "mv1", "mv2",
        "spatracker-d4", "mv1-d4", "mv2-d4",
        "mv4-b01", "mv4-b02", "mv4-b03",
        "shape-of-motion"
    ]
    assert all(m in MODELS for m in models_to_report)
    for views in ["-views1_7_14_20", "-views27_16_14_8", "-views1_4_7_11"]:
        df, csv_str = create_table({
            MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=f"panoptic-multiview{views}")
            for m in models_to_report
        }, remap=REMAP_PANOPTIC)
        print(f"*** Views: {views} ***")
        print(csv_str)
        print()


def kubric_single():
    print("Kubric single-point evaluation results:")
    print("==========================")
    for models_to_report, depths in [
        (["copycat", "cotracker3", "spatracker-pretrained",
          "spatracker", "mv1", "mv2",
          "spatracker-d4", "mv1-d4", "mv2-d4"], "-views0123"),
        (["copycat", "cotracker3",
          # "spatracker-pretrained",
          # "spatracker",
          "mv1", "mv2",
          # "spatracker-d4",
          "mv1-d4", "mv2-d4", ], "-duster0123"),
        (["cotracker3", "spatracker-pretrained",
          "spatracker", "mv1", "mv2",
          "spatracker-d4", "mv1-d4", "mv2-d4", ], "-duster0123cleaned"),
    ]:
        assert all(m in MODELS for m in models_to_report)
        df, csv_str = create_table({
            MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=f"kubric-multiview-v3{depths}")
            for m in models_to_report
        }, remap=REMAP_KUBRIC)
        print(f"Depths: {depths}")
        print(csv_str)
        print()


def dexycb_single():
    print("DexYCB single-point evaluation results:")
    print("==========================")
    for models_to_report, depths in [
        (["copycat", "cotracker3", "spatracker-pretrained",
          "spatracker", "mv1", "mv2",
          "spatracker-d4", "mv1-d4", "mv2-d4"], ""),
        (["copycat", "cotracker3", "spatracker-pretrained",
          "spatracker", "mv1", "mv2",
          "spatracker-d4", "mv1-d4", "mv2-d4", ], "-duster0123"),
        (["copycat", "cotracker3",
          # "spatracker-pretrained",
          # "spatracker",
          "mv1", "mv2",
          # "spatracker-d4",
          "mv1-d4", "mv2-d4", ], "-duster0123cleaned"),
    ]:
        assert all(m in MODELS for m in models_to_report)
        df, csv_str = create_table({
            MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=f"dex-ycb-multiview{depths}-single")
            for m in models_to_report
        }, remap=REMAP_DEXYCB_V2)
        print(f"Depths: {depths}")
        print(csv_str)
        print()


def panoptic_single():
    print("Panoptic Studio single-point evaluation results:")
    print("================================================")
    models_to_report = [
        "copycat", "cotracker3", "spatracker-pretrained",
        "spatracker", "mv1",
        "mv2",
        "spatracker-d4", "mv1-d4", "mv2-d4",
    ]
    assert all(m in MODELS for m in models_to_report)
    for views in [
        # "-views27_16_14_8",
        # "-views1_4_7_11",
        "-views1_7_14_20",
    ]:
        df, csv_str = create_table({
            MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=f"panoptic-multiview{views}-single")
            for m in models_to_report
        }, remap=REMAP_PANOPTIC)
        print(f"*** Views: {views} ***")
        print(csv_str)
        print()


MODEL_KEYS_ABLATION = [
    "copycat",
    "locotrack", "scenetracker", "delta",
    "cotracker1online", "cotracker2online", "cotracker3online",
    "cotracker1offline", "cotracker2offline", "cotracker3offline",
    "spatracker-pretrained",
    "spatracker", "mv1", "mv2",
    "spatracker-d4", "mv1-d4", "mv2-d4",
    "mv4-b01", "mv4-b02", "mv4-b03",
]


def ablation_2dpt():
    datasets = [
        "kubric-multiview-v3-views0123-2dpt",
        "kubric-multiview-v3-duster0123-2dpt",
        "dex-ycb-multiview-2dpt",
        "dex-ycb-multiview-duster0123-2dpt",
        "panoptic-multiview-views1_7_14_20-2dpt",
        "panoptic-multiview-views27_16_14_8-2dpt",
        "panoptic-multiview-views1_4_7_11-2dpt",
    ]
    models_to_report = MODEL_KEYS_ABLATION
    assert all(m in MODELS for m in models_to_report)
    for dataset in datasets:
        df, csv_str = create_table({
            MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=dataset)
            for m in models_to_report
        }, remap=REMAP_KUBRIC | PARTIAL_REMAP_FOR_2DPT_ABLATION, header=dataset == datasets[0])
        print(f"DATASET: {dataset}")
        print(csv_str)
        print()


def one_to_rule_them_all(models, datasets, separate_datasets=True, **create_table_kwargs):
    assert all(m in MODELS for m in models)
    if not separate_datasets:
        df, csv_str = create_table({
            (MODELS[m]["name"], dataset): MODELS[m]["csv"].format(dataset=dataset)
            for m in models
            for dataset in datasets
        }, remap=ONE_REMAP_TO_RULE_THEM_ALL, header=True, **create_table_kwargs)
        print(csv_str)
        print()

    else:
        for dataset in datasets:
            df, csv_str = create_table({
                MODELS[m]["name"]: MODELS[m]["csv"].format(dataset=dataset)
                for m in models
            }, remap=ONE_REMAP_TO_RULE_THEM_ALL, header=dataset == datasets[0], **create_table_kwargs)
            print(f"DATASET: {dataset}")
            print(csv_str)
            print()


def ablation_model_params():
    datasets = [
        "kubric-multiview-v3-views0123",
        "kubric-multiview-v3-duster0123",
        "dex-ycb-multiview",
        "dex-ycb-multiview-duster0123",
        "panoptic-multiview-views1_7_14_20",
        "panoptic-multiview-views27_16_14_8",
        "panoptic-multiview-views1_4_7_11",
    ]
    models = [m for m in MODELS if m.startswith("C01") or m.startswith("C02")]
    one_to_rule_them_all(models, datasets)


def ablation_camera_setups():
    datasets = [
        "panoptic-multiview-views1_7_14_20",
        "panoptic-multiview-views27_16_14_8",
        "panoptic-multiview-views1_4_7_11",
        "dex-ycb-multiview-duster0123",
        "dex-ycb-multiview-duster2345",
        "dex-ycb-multiview-duster4567",
    ]
    one_to_rule_them_all(MODEL_KEYS_ABLATION, datasets)


def ablation_num_views(separate_datasets):
    datasets = [
        "kubric-multiview-v3-views0",
        "kubric-multiview-v3-views01",
        "kubric-multiview-v3-views012",
        "kubric-multiview-v3-views0123",
        "kubric-multiview-v3-views01234",
        "kubric-multiview-v3-views012345",
        "kubric-multiview-v3-views0123456",
        "kubric-multiview-v3-views01234567",
        "kubric-multiview-v3-duster0123-views0",
        "kubric-multiview-v3-duster0123-views01",
        "kubric-multiview-v3-duster0123-views012",
        "kubric-multiview-v3-duster0123-views0123",
        "kubric-multiview-v3-duster01234567-views01234",
        "kubric-multiview-v3-duster01234567-views012345",
        "kubric-multiview-v3-duster01234567-views0123456",
        "kubric-multiview-v3-duster01234567-views01234567",
        "panoptic-multiview-views1",
        "panoptic-multiview-views1_14",
        "panoptic-multiview-views1_7_14",
        "panoptic-multiview-views1_7_14_20",
        "panoptic-multiview-views1_4_7_14_20",
        "panoptic-multiview-views1_4_7_14_17_20",
        "panoptic-multiview-views1_4_7_11_14_17_20",
        "panoptic-multiview-views1_4_7_11_14_17_20_23",
        "dex-ycb-multiview-duster0123-views0",
        "dex-ycb-multiview-duster0123-views01",
        "dex-ycb-multiview-duster0123-views012",
        "dex-ycb-multiview-duster0123-views0123",
        "dex-ycb-multiview-duster01234567-views01234",
        "dex-ycb-multiview-duster01234567-views012345",
        "dex-ycb-multiview-duster01234567-views0123456",
        "dex-ycb-multiview-duster01234567-views01234567",
    ]
    one_to_rule_them_all(MODEL_KEYS_ABLATION, datasets, separate_datasets=separate_datasets, skip_missing=True)


if __name__ == '__main__':
    # kubric_single_point()
    # kubric_before_gt0123()
    # kubric()
    # kubric_duster()

    # mv3_kubric_duster_transformed()
    # mv3_kubric_nviews()
    # mv3_kubric_duster_nviews()

    # kubric_nviews()

    # tavid2d_davis()

    # dexycb()
    # kubric_refactored()
    # panoptic()

    # kubric_single()
    # dexycb_single()
    # panoptic_single()

    # ablation_model_params()

    # ablation_2dpt()
    # ablation_camera_setups()
    # ablation_num_views(separate_datasets=False)
    # ablation_num_views(separate_datasets=True)

    #########################################

    # print("Dirty results:")
    # print("==========================")
    # df, csv_str = create_table({
    #     "CoTracker3 Online": "logs/eval/cotracker3_online/eval_tapvid2d-davis-megasam-256x256/step--1_metrics_avg.csv",
    #     "MV-Tracker + MoGe": "logs/mvtracker-may/eval_tapvid2d-davis-moge-256x256/step--1_metrics_avg.csv",
    #     "MV-Tracker + MoGe-with-extrinsics": "logs/mvtracker-may/eval_tapvid2d-davis-mogewithextrinsics-256x256/step--1_metrics_avg.csv",
    #     "MV-Tracker + ZoeDepth": "logs/mvtracker-may/eval_tapvid2d-davis-zoedepth-256x256/step--1_metrics_avg.csv",
    #     "MV-Tracker + MegaSAM": "logs/mvtracker-may/eval_tapvid2d-davis-megasam-256x256/step--1_metrics_avg.csv",
    # }, remap=REMAP_TAPVID2D, remap_index_names=REMAP_TAPVID2D_INDEX_NAMES)
    # print(csv_str)
    #
    # print("Depth + Gaussian noise")
    # print("==========================")
    # df, csv_str = create_table({
    #     f"{model};{noise}": f"{model}/eval_kubric-multiview-v3-noise{noise}/step--1_metrics_avg.csv"
    #     for model in [
    #         "logs/eval/delta",
    #         "logs/eval/spatracker_monocular_pretrained",
    #         "logs/eval/spatracker_monocular_kubric-training",
    #         "logs/eval/spatracker_monocular_duster-training",
    #         "logs/eval/spatracker_multiview_kubric-training",
    #         # "logs/eval/spatracker_multiview_duster-training",
    #         # "logs/mvtracker-noise2",
    #         "logs/eval/spatracker_multiview_duster-training-noise3",
    #         "logs/mvtracker-noise3",
    #     ]
    #     for noise in ["0cm", "1cm", "2cm", "5cm", "10cm", "20cm", "50cm", "100cm", "200cm", "1000cm"]
    # }, remap=ONE_REMAP_TO_RULE_THEM_ALL, remap_index_names=REMAP_TAPVID2D_INDEX_NAMES)
    # print(csv_str)

    #########################################

    print("Final full-scale model re-training (June 2025)")
    print("==========================")
    datasets = [
        "kubric-multiview-v3-views0123",
        "kubric-multiview-v3-duster0123",
        "dex-ycb-multiview",
        "dex-ycb-multiview-duster0123",
        "panoptic-multiview-views1_7_14_20",
        "panoptic-multiview-views27_16_14_8",
        "panoptic-multiview-views1_4_7_11",
        "tapvid2d-davis-mogewithextrinsics-256x256",
        "tapvid2d-davis-megasam-256x256",
    ]
    models = ["mvtracker-march", "mvtracker-june"]
    one_to_rule_them_all(models, datasets)

