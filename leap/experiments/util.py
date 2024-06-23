import logging
import os.path
from time import time

import numpy as np

import leap
from leap.models.base import ClassifierAccuracyPrediction


def fit_or_switch(method: ClassifierAccuracyPrediction, V, acc_fn, is_fit):
    if hasattr(method, "switch"):
        method, t_train = method.switch(acc_fn), None
        if not is_fit:
            tinit = time()
            method.fit(V)
            t_train = time() - tinit
        return method, t_train
    elif hasattr(method, "switch_and_fit"):
        tinit = time()
        method = method.switch_and_fit(acc_fn, V)
        t_train = time() - tinit
        return method, t_train
    else:
        ValueError("invalid method")


def get_predictions(method: ClassifierAccuracyPrediction, test_prot, oracle=False):
    tinit = time()
    if not oracle:
        estim_accs = method.batch_predict(test_prot)
    else:
        oracles = [Ui.prevalence() for Ui in test_prot()]
        estim_accs = method.batch_predict(test_prot, oracle_prevs=oracles)
    t_test_ave = (time() - tinit) / test_prot.total()
    return estim_accs, t_test_ave


def get_plain_prev(prev: np.ndarray):
    if prev.shape[0] > 2:
        return tuple(prev[1:])
    else:
        return prev[-1]


def prevs_from_prot(prot):
    return [get_plain_prev(Ui.prevalence()) for Ui in prot()]


def get_logger(id="quacc"):
    _name = f"{id}_log"
    _path = os.path.join(leap.env["OUT_DIR"], f"{id}.log")
    logger = logging.getLogger(_name)
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) == 0:
        fh = logging.FileHandler(_path)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%b %d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
