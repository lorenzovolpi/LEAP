import itertools as IT
import os

import numpy as np
import seaborn as sns

from exp.config import PROBLEM, get_acc_names, get_classifier_names, get_dataset_names, root_dir
from exp.util import load_results, rename_datasets, rename_methods
from leap.plot.seaborn import plot_diagonal_grid

method_map = {
    "LEAP(KDEy-MLP)": "LEAP$_{\\mathrm{KDEy}}$",
    "PHD(KDEy-MLP)": "LEAP(PPS)$_{\\mathrm{KDEy}}$",
    "OCE(KDEy-MLP)-SLSQP": "OLEAP$_{\\mathrm{KDEy}}$",
    "LEAP(CC-MLP)": "LEAP$_{\\mathrm{CC}}$",
    "LEAP(oracle)": "LEAP$_{\\Phi}$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def get_selection_datasets():
    if PROBLEM == "binary":
        return ["haberman", "pageblocks.5", "iris.2", "yeast"]
    elif PROBLEM == "multiclass":
        return ["phishing", "page_block", "academic-success", "mhr"]


def get_palette():
    base_palette = sns.color_palette("bright", 10)
    _dict = {
        "DoC": 0,
        "LEAP(KDEy-MLP)": 1,
        "PHD(KDEy-MLP)": 2,
        "OCE(KDEy-MLP)-SLSQP": 3,
        "OCE(CC-MLP)-SLSQP": 4,
        "OCE(oracle)-SLSQP": 8,
    }
    return {method_map.get(k, k): base_palette[v] for k, v in _dict.items()}


def plots():
    res = load_results()

    classifiers = get_classifier_names()
    accs = get_acc_names()
    main_methods = ["DoC", "LEAP(KDEy-MLP)", "PHD(KDEy-MLP)", "OCE(KDEy-MLP)-SLSQP"]
    oracle_methods = ["OCE(CC-MLP)-SLSQP", "OCE(KDEy-MLP)-SLSQP", "OCE(oracle)-SLSQP"]

    configs = [
        {
            "name": "all",
            "datasets": get_dataset_names(),
            "methods": main_methods,
        },
        {
            "name": "4x1",
            "datasets": get_selection_datasets(),
            "methods": main_methods,
        },
        {
            "name": "all_oracle",
            "datasets": get_dataset_names(),
            "methods": oracle_methods,
        },
        {
            "name": "4x1_oracle",
            "datasets": get_selection_datasets(),
            "methods": oracle_methods,
        },
    ]

    parent_dir = os.path.join(root_dir, "plots")
    os.makedirs(parent_dir, exist_ok=True)

    for cls_name, acc in IT.product(classifiers[:1], accs):
        print(f"Plotting {cls_name} for {acc}")
        df = res.loc[(res["classifier"] == cls_name) & (res["acc_name"] == acc), :]

        for cfg in configs:
            name, datasets, methods = cfg["name"], cfg["datasets"], cfg["methods"]
            dfc = df.loc[(df["dataset"].isin(datasets)) & (df["method"].isin(methods)), :]
            dfc, methods = rename_methods(method_map, dfc, methods)
            dfc, datasets = rename_datasets(dataset_map, dfc, datasets)

            plot_diagonal_grid(
                dfc,
                methods,
                basedir=parent_dir,
                filename=f"grid_{cls_name}_{PROBLEM}_{name}",
                n_cols=4,
                legend_bbox_to_anchor=(0.95, 0.3),
                # palette="deep",
                # palette=sns.color_palette("hls", len(_methods)),
                palette=get_palette(),
                x_label="True Accuracy",
                y_label="Estimated Accuracy",
                aspect=0.8,
                xtick_vert=True,
                xticks=np.linspace(0, 1, 6, endpoint=True),
                yticks=np.linspace(0, 1, 6, endpoint=True),
            )


if __name__ == "__main__":
    plots()
