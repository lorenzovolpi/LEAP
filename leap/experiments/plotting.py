import itertools as IT

import numpy as np
import pandas as pd

from leap.experiments.generators import gen_acc_measure, gen_bin_datasets
from leap.experiments.report import Report
from leap.plot.seaborn import plot_diagonal_grid

PROBLEM = "binary"
basedir = PROBLEM
plots_basedir = PROBLEM

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets


def rename_methods(df: pd.DataFrame, methods: dict):
    for om, nm in methods.items():
        df.loc[df["method"] == om, "method"] = nm


def plot_grid_of_diagonals(
    methods, dataset_names, classifiers, filename=None, n_cols=5, **kwargs
):
    for cls_name, (acc_name, _) in IT.product(classifiers, gen_acc_measure()):
        rep = Report.load_results(
            basedir,
            cls_name,
            acc_name,
            dataset_name=dataset_names,
            method_name=list(methods.keys()),
        )
        df = rep.table_data(mean=False)
        rename_methods(df, methods)
        plot_diagonal_grid(
            df,
            cls_name,
            acc_name,
            dataset_names,
            basedir=plots_basedir,
            n_cols=n_cols,
            x_label="True Accuracy",
            y_label="Estimated Accuracy",
            file_name=f"{PROBLEM}_{filename}" if filename else PROBLEM,
            **kwargs,
        )
        print(f"{cls_name}-{acc_name} plots generated")


if __name__ == "__main__":
    methods = {
        "ATC": "ATC",
        "DoC": "DoC",
        "LEAP": "LEAP",
        "LEAP-plus": "LEAP$^+$",
    }
    selected_datasets = ["sonar", "haberman", "cmc.2", "german", "iris.2"]
    plot_grid_of_diagonals(
        methods,
        selected_datasets,
        ["LR"],
        filename="5x1",
        n_cols=5,
        legend_bbox_to_anchor=(0.96, 0.3),
        legend_wspace=0.08,
        xtick_vert=True,
        aspect=0.8,
        xticks=np.linspace(0, 1, 6, endpoint=True),
        yticks=np.linspace(0, 1, 6, endpoint=True),
    )

    all_datasets = [name for name, _ in gen_datasets(only_names=True)]
    classifiers = ["LR", "KNN", "SVM", "MLP"]
    plot_grid_of_diagonals(
        methods,
        all_datasets,
        classifiers,
        filename="all",
        n_cols=5,
        legend_bbox_to_anchor=(0.84, 0.06),
    )
