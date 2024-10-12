import os
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from leap.utils.commons import get_plots_path

sns.set_theme(style="whitegrid")

DPI = 300


def _save_figure(plot: Axes, cls_name, plot_type, filename):
    exts = [
        "svg",
        "png",
    ]
    files = [get_plots_path(cls_name, plot_type, filename, ext=e) for e in exts]
    for f in files:
        os.makedirs(Path(f).parent, exist_ok=True)
        plot.figure.savefig(f, bbox_inches="tight", dpi=DPI)
    plot.figure.clear()


def _config_legend(plot: Axes, bbox_to_anchor):
    sns.move_legend(plot, "lower center", bbox_to_anchor=bbox_to_anchor, ncol=1)


def plot_diagonal(
    df: pd.DataFrame,
    cls_name,
    acc_name,
    dataset_name,
    *,
    basedir=None,
    filename="diagonal",
    color_palette=None,
    legend_bbox_to_anchor=(1, 0.5),
    legend_label_length=8,
    x_label="true accs.",
    y_label="estim. accs.",
):
    if color_palette is not None:
        palette = sns.color_palette(color_palette)
    else:
        palette = sns.color_palette()

    plot = sns.scatterplot(
        data=df,
        x="true_accs",
        y="estim_accs",
        hue="method",
        alpha=0.5,
        palette=palette,
    )
    plot.set_xlim((0, 1))
    plot.set_ylim((0, 1))
    plot.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)
    plot.set_aspect(1.0)

    handles, labels = plot.get_legend_handles_labels()
    for lh in handles:
        lh.set_alpha(1)
    t_labels = []
    for lbl in labels:
        if len(lbl) < legend_label_length:
            t_labels.append(lbl + " " * (legend_label_length - len(lbl)))
    plot.legend(handles, t_labels, title="")
    _config_legend(plot, legend_bbox_to_anchor)

    plot.set_xlabel(x_label)
    plot.set_ylabel(y_label)

    return _save_figure(plot, cls_name, dataset_name, filename)


def plot_diagonal_grid(
    df: pd.DataFrame,
    cls_name,
    acc_name,
    dataset_names,
    *,
    basedir=None,
    file_name="diagonal",
    color_palette=None,
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
    **kwargs,
):
    if color_palette is not None:
        palette = sns.color_palette(color_palette)
    else:
        palette = sns.color_palette()

    plot = sns.FacetGrid(
        df,
        col="dataset",
        col_wrap=n_cols,
        hue="method",
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

    return _save_figure(plot, cls_name, "grid", file_name)
