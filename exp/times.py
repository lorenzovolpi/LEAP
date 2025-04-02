import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, MultipleLocator

from exp.critdd import get_acc_names
from exp.leap.config import PROBLEM, get_classifier_names, get_dataset_names, root_dir
from exp.leap.util import load_results, rename_methods

method_map = {
    "Naive": 'Na\\"ive',
    "ATC-MC": "ATC",
    "LEAP(KDEy)": "LEAP$_{\\mathrm{KDEy}}$",
    "S-LEAP(KDEy)": "S-LEAP$_{\\mathrm{KDEy}}$",
    "O-LEAP(KDEy)": "O-LEAP$_{\\mathrm{KDEy}}$",
}


def _get_n_classes(train_prev):
    if isinstance(train_prev, list):
        return len(train_prev) + 1
    elif isinstance(train_prev, float):
        return 2
    else:
        return 0


def pivot(df, parent_dir):
    pivot = pd.pivot_table(df, index="n_classes", columns="method", values=["t_train", "t_test_ave"])
    with open(os.path.join(parent_dir, f"time_table_{PROBLEM}.html"), "w") as f:
        pivot.to_html(f)


def plot(df, methods, parent_dir):
    exts = ["png", "pdf"]
    paths = [os.path.join(parent_dir, f"time_plot_{PROBLEM}.{ext}") for ext in exts]

    plot = sns.lineplot(
        data=df,
        x="n_classes",
        y="t_test_ave",
        hue="method",
        hue_order=methods,
        errorbar="sd",
        err_style="bars",
    )
    plot.legend(title="")
    plot.set_xlabel("Number of classes ($n$)")
    plot.set_ylabel("Avg. time log (s)")
    plot.set(yscale="log")
    plot.xaxis.set_major_locator(MultipleLocator(2, offset=df["n_classes"].min()))
    plot.yaxis.set_major_locator(MultipleLocator(1))
    plot.yaxis.set_minor_locator(LogLocator(10))
    for p in paths:
        plot.figure.savefig(p)
    plot.figure.clear()
    plt.close(plot.figure)


def times():
    classifiers = get_classifier_names()
    accs = get_acc_names()
    datasets = get_dataset_names()
    methods = ["ATC-MC", "DoC", "LEAP(KDEy)", "S-LEAP(KDEy)", "O-LEAP(KDEy)"]

    res = load_results(filter_methods=methods)

    parent_dir = os.path.join(root_dir, "times")
    os.makedirs(parent_dir, exist_ok=True)

    for acc in accs:
        df = res.loc[
            (res["classifier"].isin(classifiers))
            & (res["acc_name"] == acc)
            & (res["dataset"].isin(datasets))
            & (res["method"].isin(methods)),
            ["t_train", "t_test_ave", "method", "dataset", "train_prev"],
        ]
        df["n_classes"] = df["train_prev"].map(_get_n_classes)
        df, _methods = rename_methods(method_map, df, methods)

        plot(df, _methods, parent_dir)


if __name__ == "__main__":
    times()
