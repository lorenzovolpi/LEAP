import itertools as IT
import os

import numpy as np
import seaborn as sns
from quacc.plot.seaborn import plot_diagonal_grid

from exp.leap.config import PROBLEM, get_acc_names, get_classifier_names, get_dataset_names, root_dir
from exp.leap.util import load_results, rename_datasets, rename_methods

method_map = {
    "LEAP(KDEy)": "LEAP$_{\\mathrm{KDEy}}$",
    "S-LEAP(KDEy)": "S-LEAP$_{\\mathrm{KDEy}}$",
    "O-LEAP(KDEy)": "O-LEAP$_{\\mathrm{KDEy}}$",
    "O-LEAP(CC)": "O-LEAP$_{\\mathrm{CC}}$",
    "O-LEAP(oracle)": "O-LEAP$_{\\Phi}$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def get_selection_datasets():
    if PROBLEM == "binary":
        return ["pageblocks.5", "yeast", "haberman", "iris.2"]
    elif PROBLEM == "multiclass":
        return ["poker_hand", "shuttle", "page_block", "phishing"]


def get_palette():
    base_palette = sns.color_palette("bright", 10)
    _dict = {
        "DoC": 0,
        "LEAP(KDEy)": 1,
        "S-LEAP(KDEy)": 2,
        "O-LEAP(KDEy)": 3,
        "O-LEAP(CC)": 4,
        "O-LEAP(oracle)": 8,
    }
    return {method_map.get(k, k): base_palette[v] for k, v in _dict.items()}


def plots():
    all_classifiers = get_classifier_names()
    accs = get_acc_names()
    main_methods = ["DoC", "LEAP(KDEy)", "S-LEAP(KDEy)", "O-LEAP(KDEy)"]
    oracle_methods = ["O-LEAP(CC)", "O-LEAP(KDEy)", "O-LEAP(oracle)"]

    res = load_results(filter_methods=oracle_methods + main_methods)

    configs = [
        {
            "name": "all",
            "datasets": get_dataset_names(),
            "methods": main_methods,
            "classifiers": all_classifiers,
        },
        {
            "name": "4x1",
            "datasets": get_selection_datasets(),
            "methods": main_methods,
            "classifiers": ["LR"],
        },
        {
            "name": "all_oracle",
            "datasets": get_dataset_names(),
            "methods": oracle_methods,
            "classifiers": all_classifiers,
        },
        {
            "name": "4x1_oracle",
            "datasets": get_selection_datasets(),
            "methods": oracle_methods,
            "classifiers": ["LR"],
        },
    ]

    parent_dir = os.path.join(root_dir, "plots")
    os.makedirs(parent_dir, exist_ok=True)

    for cfg, acc in IT.product(configs, accs):
        name, classifiers, datasets, methods = cfg["name"], cfg["classifiers"], cfg["datasets"], cfg["methods"]
        for cls_name in classifiers:
            df = res.loc[
                (res["dataset"].isin(datasets))
                & (res["method"].isin(methods))
                & (res["acc_name"] == acc)
                & (res["classifier"] == cls_name),
                :,
            ]
            print(f"Plotting {name} for {cls_name} [{acc}]")
            assert len(df["method"].unique()) == len(methods), (
                f"Error while generating {name} for {cls_name} [{acc}]: some methods missing!"
            )
            _df, _methods = rename_methods(method_map, df, methods)
            _df, _datasets = rename_datasets(dataset_map, df, datasets)

            plot_diagonal_grid(
                _df,
                methods_order=_methods,
                datasets_order=_datasets,
                basedir=parent_dir,
                filename=f"grid_{cls_name}_{PROBLEM}_{name}",
                n_cols=4,
                legend_bbox_to_anchor=(0.95, 0.3),
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
