import itertools as IT

import numpy as np
import pandas as pd

from leap.experiments.generators import gen_acc_measure, gen_bin_datasets
from leap.experiments.report import Report
from leap.plot.seaborn import plot_diagonal, plot_diagonal_grid

PROBLEM = "binary"
basedir = PROBLEM
plots_basedir = PROBLEM

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets


def rename_methods(df: pd.DataFrame, methods: dict):
    for om, nm in methods.items():
        df.loc[df["method"] == om, "method"] = nm


def filter_methods(df: pd.DataFrame, methods_list: list) -> pd.DataFrame:
    mask = df["method"].isin(methods_list)
    return df[mask]


def plot_diagonals(methods, dataset_name, acc_name, classifier, color_palette, **kwargs):
    rep = Report.load_results(
        basedir,
        classifier,
        acc_name,
        dataset_name=[dataset_name],
        method_name=list(methods.keys()),
    )
    df = rep.table_data(mean=False)
    for i in range(len(methods)):
        methods_list = list(methods.keys())[: i + 1]
        _df = filter_methods(df, methods_list)
        rename_methods(_df, methods)
        plot_diagonal(
            _df,
            classifier,
            acc_name,
            dataset_name,
            basedir=plots_basedir,
            filename=f"{PROBLEM}_{i+1}",
            color_palette=color_palette[: i + 1],
            x_label="True Accuracy",
            y_label="Estimated Accuracy",
            **kwargs,
        )
    print("plots generated")


if __name__ == "__main__":
    methods_dict = {
        "ATC": "ATC",
        "DoC": "DoC",
        "LEAP": "LEAP",
        "LEAP-plus": "LEAP$+$",
    }
    color_palette = {
        "ATC": "#1f77b4",
        "DoC": "#ff7f0e",
        "LEAP": "#2ca02c",
        "LEAP-plus": "#d62728",
    }
    methods = ["ATC", "DoC", "LEAP", "LEAP-plus"]
    dataset_name = "cmc.2"
    acc_name = "vanilla_accuracy"
    classifier = "LR"
    plot_diagonals(
        {k: methods_dict[k] for k in methods},
        dataset_name,
        acc_name,
        classifier,
        [color_palette[m] for m in methods],
        legend_bbox_to_anchor=(1.15, 0.4),
    )
