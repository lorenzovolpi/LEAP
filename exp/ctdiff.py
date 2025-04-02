import itertools as IT
import math
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
import numpy as np
import pandas as pd
import seaborn as sns

from exp.config import PROBLEM, get_acc_names, get_classifier_names, get_dataset_names, root_dir
from exp.util import load_results, rename_datasets, rename_methods

N_COLS = 4

method_map = {
    "LEAP(ACC)": "LEAP$_{\\mathrm{ACC}}$",
    "LEAP(KDEy)": "LEAP$_{\\mathrm{KDEy}}$",
    "S-LEAP(KDEy)": "S-LEAP$_{\\mathrm{KDEy}}$",
    "O-LEAP(KDEy)": "O-LEAP$_{\\mathrm{KDEy}}$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def _savefig(plot, path):
    exts = ["png", "pdf"]
    paths = [f"{path}.{ext}" for ext in exts]
    for p in paths:
        plot.figure.savefig(p)
    plot.figure.clear()
    plt.close(plot.figure)


def get_cts(df, method, cts_name):
    return np.array(df.loc[df["method"] == method, cts_name].to_list())


def draw_heatmap(data, plot_names, **kwargs):
    col = data["col"].to_numpy()[0]
    row = data["row"].to_numpy()[0]
    plot_name = plot_names[col + N_COLS * row]
    data = data.drop(["col", "row"], axis=1).to_numpy()
    plot = sns.heatmap(data, **kwargs)
    plot.set_title(plot_name)


def ctdfiff():
    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    methods = ["Naive", "LEAP(KDEy)", "S-LEAP(KDEy)", "O-LEAP(KDEy)"]

    res = load_results(filter_methods=methods)

    parent_dir = os.path.join(root_dir, "ctdiffs")
    os.makedirs(parent_dir, exist_ok=True)

    res, datasets = rename_datasets(dataset_map, res, datasets)
    res, methods = rename_methods(method_map, res, methods)

    for cls_name, acc, dataset in IT.product(classifiers, accs, datasets):
        print(cls_name, acc, dataset)
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc) & (res["dataset"] == dataset), :]

        cnt = 0

        plot_names, cbars = [], []
        vmin, vmax = 1, 0
        annot = False

        mdfs = []
        for m in methods:
            true_cts = get_cts(df, m, "true_cts")
            estim_cts = get_cts(df, m, "estim_cts")
            _ae = np.abs(estim_cts - true_cts).mean(axis=0)
            sqae = np.sqrt(_ae)

            mdf = pd.DataFrame(sqae)
            mdf["col"] = cnt % N_COLS
            mdf["row"] = 0
            mdfs.append(mdf)
            plot_names.append(m)
            cbars.append(cnt == len(methods) - 1)
            ae_min, ae_max = np.min(sqae), np.max(sqae)
            vmin = ae_min if ae_min < vmin else vmin
            vmax = ae_max if ae_max > vmax else vmax
            annot = sqae.shape[1] <= 4
            cnt += 1

        hmdf = pd.concat(mdfs, axis=0)
        plot = sns.FacetGrid(hmdf, col="col", row="row")
        cbar_ax = plot.fig.add_axes([0.92, 0.15, 0.02, 0.7])
        plot = plot.map_dataframe(
            draw_heatmap,
            plot_names=plot_names,
            vmin=vmin,
            vmax=vmax,
            cmap="rocket_r",
            annot=annot,
            cbar_ax=cbar_ax,
        )
        plot.fig.subplots_adjust(right=0.9)
        for ax in plot.axes.flatten():
            formatter = tkr.FuncFormatter(lambda x, p: "$\\omega_{" + f"{int(math.floor(x) + 1)}" + "}$")
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

        path = os.path.join(parent_dir, f"heatmap_{cls_name}_{dataset}_{PROBLEM}")
        _savefig(plot, path)


if __name__ == "__main__":
    ctdfiff()
