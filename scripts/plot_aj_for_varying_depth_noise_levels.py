import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# set_size from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1, golden_ratio=(5 ** .5 - 1) / 2):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    # golden_ratio = (5 ** .5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def setup_plot():
    sns.set_theme(style="whitegrid")
    sns.set_palette("tab10")
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.weight'] = 'normal'


def plot_aj(
        save_name='plot_robustness_to_depth_noise.pdf',
        width_in_paper_pts=237.13594,  # \showthe\linewidth --> > 237.13594pt.
        linewidth=1.5,
        marker_size=5,
        label_font_size=9,
        tick_font_size=9,
        legend_font_size=7,
        dpi=400,
        results_dir=None,
        save_svg=False,
):
    setup_plot()

    fig, ax = plt.subplots(figsize=set_size(width_in_paper_pts, golden_ratio=0.3), dpi=dpi)

    x_labels = ['0', '1', '2', '5', '10', '20', '50', '100', '200']
    x = np.arange(len(x_labels))
    # x = np.array([0, 1, 2, 5, 10, 20, 50, 100, 200])
    # x_labels = ['0', '1', '2', '5', '10', '20', '50', '100', '200']

    results = {
        "Ours": [81.6, 80.7, 77.4, 69.8, 63.1, 59.3, 56.1, 54.3, 52.8],
        "Triplane": [75.4, 75.0, 73.7, 69.2, 63.4, 57.4, 51.5, 49.1, 47.6],
        "SpaTracker": [65.5, 63.8, 62.1, 58.7, 55.8, 52.6, 48.6, 45.4, 43.3],
        "DELTA": [57.4, 51.8, 46.2, 34.3, 23.8, 13.2, 5.0, 2.3, 1.0],
    }

    for label, y in results.items():
        sns.lineplot(x=x, y=y, ax=ax, linewidth=linewidth, marker='o', markersize=marker_size, label=label)
    # ax.axhline(y=47.2, color=sns.color_palette("tab10")[1], linestyle='--', linewidth=1.5, label='Blind Baseline')

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(40, 90, 10))
    ax.set_ylim([40, 83])
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

    ax.set_xlabel('Depth Noise (σ, in cm)', fontsize=label_font_size, fontweight='normal', labelpad=0)
    ax.set_ylabel('AJ', fontsize=label_font_size, fontweight='normal', labelpad=2)

    for spine in ax.spines.values():
        spine.set_color('black')

    ax.grid(axis='y', color='lightgrey')

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    legend = plt.legend(
        frameon=True,
        fancybox=False,
        loc=(0.675, 0.265),
        # loc="upper right",
        prop={'size': legend_font_size},
        handletextpad=0.2,
        labelspacing=0.1,
    )
    # legend.get_frame().set_facecolor('white')
    # legend.get_frame().set_edgecolor('black')

    plt.tight_layout(pad=0)

    if save_name:
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
            save_name = os.path.join(results_dir, save_name)

        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        if save_svg:
            plt.savefig(save_name.replace('.pdf', '.svg'), bbox_inches='tight', pad_inches=0)

    plt.show()


if __name__ == '__main__':
    plot_aj()
