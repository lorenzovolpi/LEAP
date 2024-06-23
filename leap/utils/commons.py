import json
import os
from pathlib import Path
from typing import Callable

import quapy as qp
from quapy.data.base import LabelledCollection
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

import leap


def save_json_file(path, data):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_json_file(path, object_hook=None):
    if not os.path.exists(path):
        raise ValueError("Ivalid path for json file")
    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)


def get_results_path(basedir, cls_name, acc_name, dataset_name, method_name):
    return os.path.join(
        leap.env["OUT_DIR"],
        "results",
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        method_name + ".json",
    )


def get_plots_path(cls_name, plot_type, filename, ext="svg"):
    return os.path.join(
        leap.env["OUT_DIR"], "plots", f"{cls_name}_{plot_type}_{filename}.{ext}"
    )


def true_acc(h: BaseEstimator, acc_fn: Callable, U: LabelledCollection):
    y_pred = h.predict(U.X)
    y_true = U.y
    conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=U.classes_)
    return acc_fn(conf_table)


def save_dataset_stats(dataset_name, test_prot, L, V):
    path = os.path.join(leap.env["OUT_DIR"], "dataset_stats", f"{dataset_name}.json")
    test_prevs = [Ui.prevalence() for Ui in test_prot()]
    shifts = [qp.error.ae(L.prevalence(), Ui_prev) for Ui_prev in test_prevs]
    info = {
        "n_classes": L.n_classes,
        "n_train": len(L),
        "n_val": len(V),
        "train_prev": L.prevalence().tolist(),
        "val_prev": V.prevalence().tolist(),
        "test_prevs": [x.tolist() for x in test_prevs],
        "shifts": [x.tolist() for x in shifts],
        "sample_size": test_prot.sample_size,
        "num_samples": test_prot.total(),
    }
    save_json_file(path, info)
