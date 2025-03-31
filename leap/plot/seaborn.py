import os

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

sns.set_theme(style="whitegrid")
sns.set_palette("colorblind")

DPI = 300


def _save_figure(plot: Axes, basedir, filename):
    exts = [
        # "svg",
        "pdf",
        "png",
    ]
    files = [os.path.join(basedir, f"{filename}.{ext}") for ext in exts]
    for f in files:
        plot.figure.savefig(f, bbox_inches="tight", dpi=DPI)
    plot.figure.clear()


def plot_diagonal_grid(
    df: pd.DataFrame,
    method_names,
    *,
    basedir="output",
    filename="diagonal_grid",
    n_cols=1,
    x_label="true accs.",
    y_label="estim. accs.",
    aspect=1,
    xticks=None,
    yticks=None,
    xtick_vert=False,
    hspace=0.1,
    wspace=0.1,
    legend_bbox_to_anchor=(1, 0.5),
    palette="tab10",
    **kwargs,
):
    plot = sns.FacetGrid(
        df,
        col="dataset",
        col_wrap=n_cols,
        hue="method",
        hue_order=method_names,
        xlim=(0, 1),
        ylim=(0, 1),
        aspect=aspect,
        palette=palette,
    )
    plot.map(sns.scatterplot, "true_accs", "estim_accs", alpha=0.2, s=20, edgecolor=None)
    for ax in plot.axes.flat:
        ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)
        if xtick_vert:
            ax.tick_params(axis="x", labelrotation=90, labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

    plot.figure.subplots_adjust(hspace=hspace, wspace=wspace)
    plot.set_titles("{col_name}")

    plot.add_legend(title="")
    sns.move_legend(
        plot,
        "lower center",
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=kwargs.get("legend_ncol", 1),
    )
    for lh in plot.legend.legend_handles:
        lh.set_alpha(1)
        lh.set_sizes([100])

    plot.set_xlabels(x_label)
    plot.set_ylabels(y_label)

    return _save_figure(plot=plot, basedir=basedir, filename=filename)
