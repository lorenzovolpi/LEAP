import glob
import logging
import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

import leap
from exp.config import PROBLEM, get_method_names, root_dir
from leap.models.base import ClassifierAccuracyPrediction
from leap.models.cont_table import CAPContingencyTable


def load_results(filter_methods=None) -> pd.DataFrame:
    dfs = []
    _methods = get_method_names() if filter_methods is None else filter_methods
    for path in glob.glob(os.path.join(root_dir, "data", PROBLEM, "**", "*.json"), recursive=True):
        if Path(path).stem in _methods:
            dfs.append(pd.read_json(path))

    return pd.concat(dfs, axis=0)


def rename_datasets(mapping, df, datasets):
    _datasets = [mapping.get(d, d) for d in datasets]
    for d, rd in mapping.items():
        df.loc[df["dataset"] == d, "dataset"] = rd
    return df, _datasets


def rename_methods(mapping, df, methods, baselines=None):
    _methods = [mapping.get(m, m) for m in methods]
    for m, rm in mapping.items():
        df.loc[df["method"] == m, "method"] = rm

    if baselines is None:
        return df, _methods
    else:
        _baselines = [mapping.get(b, b) for b in baselines]
        return df, _methods, _baselines


def method_can_switch(method):
    return method is not None and hasattr(method, "switch")


def fit_or_switch(method: ClassifierAccuracyPrediction, V, V_posteriors, acc_fn, is_fit):
    if hasattr(method, "switch"):
        method, t_train = method.switch(acc_fn), None
        if not is_fit:
            tinit = time()
            method.fit(V, V_posteriors)
            t_train = time() - tinit
        return method, t_train
    elif hasattr(method, "switch_and_fit"):
        tinit = time()
        method = method.switch_and_fit(acc_fn, V, V_posteriors)
        t_train = time() - tinit
        return method, t_train
    else:
        ValueError("invalid method")


def get_predictions(method: ClassifierAccuracyPrediction, test_prot, test_prot_posteriors):
    tinit = time()
    estim_accs = method.batch_predict(test_prot, test_prot_posteriors)
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, t_test_ave


def get_ct_predictions(method: ClassifierAccuracyPrediction, test_prot, test_prot_posteriors):
    tinit = time()
    if isinstance(method, CAPContingencyTable):
        estim_accs, estim_cts = method.batch_predict(test_prot, test_prot_posteriors, get_estim_cts=True)
    else:
        estim_accs, estim_cts = method.batch_predict(test_prot, test_prot_posteriors), None
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, estim_cts, t_test_ave


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return np.around(prev[1:], decimals=4).tolist()
    else:
        return float(np.around(prev, decimals=4)[-1])


def prevs_from_prot(prot):
    return [get_plain_prev(Ui.prevalence()) for Ui in prot()]


def get_acc_name(acc_name):
    return {
        "Vanilla Accuracy": "vanilla_accuracy",
        "Macro F1": "macro-F1",
    }


def get_logger(id="quacc"):
    _name = f"{id}_log"
    _path = os.path.join(leap.env["OUT_DIR"], f"{id}.log")
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        fh = logging.FileHandler(_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%b %d %H:%M:%S")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def gen_model_dataset(_gen_model, _gen_dataset):
    for model in _gen_model():
        for dataset in _gen_dataset():
            yield model, dataset


def timestamp(t_train: float, t_test_ave: float) -> str:
    t_train = round(t_train, ndigits=3)
    t_test_ave = round(t_test_ave, ndigits=3)
    return f"{t_train=}s; {t_test_ave=}s"
