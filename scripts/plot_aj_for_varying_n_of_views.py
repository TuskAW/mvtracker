import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns


# set_size from https://jwalton.info/Embed-Publication-Matplotlib-Latex/
def set_size(width, fraction=1):
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
    golden_ratio = (5 ** .5 - 1) / 2

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
        save_name='plot_number_of_views.pdf',
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

    fig, ax = plt.subplots(figsize=set_size(width_in_paper_pts), dpi=dpi)

    x = np.arange(1, 9)

    y_data = {
        "MVTracker (ours)": [64.0, 66.8, 73.2, 71.1, 77.4, 76.7, 77.3, 79.2],
        "Triplane": [44.0, 48.0, 56.0, 57.6, 63.5, 64.5, 65.5, 66.8],
        # "TAPIP3D": [36.6, 35.6, 40.5, 38.8, 57.7, 54.2, 55.2, 56.4],
        # "SpatialTrackerV2": [39.8, 39.5, 36.5, 35.5, 41.1, 37.1, 37.0, 37.7],
        "SpatialTracker": [60.6, 58.4, 61.8, 58.3, 63.2, 62.4, 62.9, 63.4],
        "CoTracker3": [28.6, 27.0, 29.5, 29.4, 39.1, 37.5, 37.1, 37.3],
        # "CoTracker2": [29.8, 26.4, 29.2, 28.8, 37.8, 36.2, 36.0, 36.0],
        "DELTA": [33.0, 34.3, 38.0, 36.5, 37.2, 35.4, 34.9, 35.7],
        "LocoTrack": [27.9, 26.0, 28.1, 27.8, 36.3, 34.8, 34.7, 34.9]
    }

    for label, y in y_data.items():
        sns.lineplot(x=x, y=y, label=label, ax=ax, linewidth=linewidth, marker='o', markersize=marker_size)

    ax.set_xticks(x)
    ax.set_yticks(np.arange(30, 81, 10))
    ax.set_ylim([25, 80])
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)

    ax.set_xlabel('Number of Views', fontsize=label_font_size, fontweight='normal', labelpad=0)
    ax.set_ylabel('Average Jaccard (AJ)', fontsize=label_font_size, fontweight='normal', labelpad=2)

    for spine in ax.spines.values():
        spine.set_color('black')

    ax.grid(axis='y', color='lightgrey')

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    legend = plt.legend(
        frameon=True,
        fancybox=False,
        loc=(0.625, 0.265),
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
