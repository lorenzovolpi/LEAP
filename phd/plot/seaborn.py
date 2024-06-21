import os
from pathlib import Path

import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from phd.utils.commons import get_plots_path

sns.set_theme(style="whitegrid")

DPI = 300


def _save_figure(plot: Axes, basedir, cls_name, acc_name, dataset_name, plot_type):
    exts = [
        "svg",
        "png",
    ]
    plotsubdir = "all" if dataset_name == "*" else dataset_name
    files = [
        get_plots_path(basedir, cls_name, acc_name, plotsubdir, plot_type, ext=e)
        for e in exts
    ]
    for f in files:
        os.makedirs(Path(f).parent, exist_ok=True)
        plot.figure.savefig(f, bbox_inches="tight", dpi=DPI)
    plot.figure.clear()


def _config_legend(plot: Axes):
    plot.legend(title="")
    sns.move_legend(plot, "lower center", bbox_to_anchor=(1, 0.5), ncol=1)


def plot_diagonal(
    df: pd.DataFrame, cls_name, acc_name, dataset_name, *, basedir=None, file_name=None
):
    plot = sns.scatterplot(
        data=df, x="true_accs", y="estim_accs", hue="method", alpha=0.5
    )

    _config_legend(plot)
    return _save_figure(
        plot,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        "diagonal" if file_name is None else file_name,
    )


def plot_diagonal_grid(
    df: pd.DataFrame,
    cls_name,
    acc_name,
    dataset_names,
    *,
    basedir=None,
    file_name=None,
    n_cols=1,
    **kwargs,
):
    plot = sns.FacetGrid(df, col="dataset", col_wrap=n_cols, hue="method")
    plot.map(
        sns.scatterplot, "true_accs", "estim_accs", alpha=0.2, s=20, edgecolor=None
    )
    for ax in plot.axes.flat:
        ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=1)

    plot.figure.subplots_adjust(hspace=0.1, wspace=0.1)
    plot.set(xlim=(0, 1), ylim=(0, 1))
    plot.set_titles("{col_name}")

    plot.add_legend(title="")
    sns.move_legend(plot, "lower center", bbox_to_anchor=(0.84, 0.06), ncol=1)
    for lh in plot.legend.legend_handles:
        lh.set_alpha(1)
        lh.set_sizes([100])

    if "x_label" in kwargs:
        plot.set_xlabels(kwargs["x_label"])
    if "y_label" in kwargs:
        plot.set_ylabels(kwargs["y_label"])

    return _save_figure(
        plot,
        basedir,
        cls_name,
        acc_name,
        "grid",
        "diagonal" if file_name is None else file_name,
    )
