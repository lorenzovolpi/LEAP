import os
from abc import ABC
from contextlib import ExitStack, contextmanager
from typing import Callable, Literal

import numpy as np
import quacc as qc
import quapy as qp
from joblib import Parallel, delayed
from quapy.data.base import LabelledCollection
from scipy.sparse import coo_array
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix


class NaNError(Exception):
    def __init__(self, dataset, method, *args: object) -> None:
        super().__init__(*args)
        self.dataset = dataset
        self.method = method


class SparseMatrixBuilder(ABC):
    def __init__(self):
        self.data = []
        self.rows = []
        self.cols = []

    def add(self, data: np.ndarray, rows: np.ndarray, cols: np.ndarray):
        assert len(data) == len(rows) == len(cols), "Inconsistent array lengths; cannot add to SparseMatrixBuilder"
        self.data.append(data)
        self.rows.append(rows)
        self.cols.append(cols)

    def build_csr(self, shape):
        data = np.hstack(self.data)
        rows = np.hstack(self.rows)
        cols = np.hstack(self.cols)
        coo = coo_array((data, (rows, cols)), shape=shape)
        return coo.tocsr()


def get_njobs(n_jobs):
    return qc.env["N_JOBS"] if n_jobs is None else n_jobs


def true_acc(h: BaseEstimator, acc_fn: Callable, U: LabelledCollection):
    y_pred = h.predict(U.X)
    y_true = U.y
    conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=U.classes_)
    return acc_fn(conf_table)


@contextmanager
def temp_force_njobs(force):
    if force:
        openblas_nt_was_set = "OPENBLAS_NUM_THREADS" in os.environ
        if openblas_nt_was_set:
            openblas_nt_old = os.getenv("OPENBLAS_NUM_THREADS")

        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        yield
    finally:
        if force:
            if openblas_nt_was_set:
                os.environ["OPENBLAS_NUM_THREADS"] = openblas_nt_old
            else:
                os.environ.pop("OPENBLAS_NUM_THREADS")


def parallel(
    func,
    args_list,
    n_jobs,
    seed=None,
    return_as: Literal["list"] | Literal["array"] | Literal["generator"] | Literal["generator_unordered"] = "list",
    backend="loky",
    verbose=0,
    batch_size="auto",
    max_nbytes="1M",
):
    """
    A wrapper of multiprocessing:

    >>> Parallel(n_jobs=n_jobs)(
    >>>      delayed(func)(args_i) for args_i in args
    >>> )

    that takes the `quapy.environ` variable as input silently.
    Seeds the child processes to ensure reproducibility when n_jobs>1.

    :param func: callable
    :param args: args of func
    :param seed: the numeric seed
    :param asarray: set to True to return a np.ndarray instead of a list
    :param backend: indicates the backend used for handling parallel works
    """

    def func_dec(qp_environ, qc_environ, seed, *args):
        qp.environ = qp_environ.copy()
        qp.environ["N_JOBS"] = 1
        qc.env = qc_environ.copy()
        qc.env["N_JOBS"] = 1
        # set a context with a temporal seed to ensure results are reproducibles in parallel
        with ExitStack() as stack:
            if seed is not None:
                stack.enter_context(qp.util.temp_seed(seed))
            return func(*args)

    _returnas = "list" if return_as == "array" else return_as
    with ExitStack() as stack:
        stack.enter_context(qc.commons.temp_force_njobs(qc.env["FORCE_NJOBS"]))
        out = Parallel(
            n_jobs=n_jobs,
            return_as=_returnas,
            backend=backend,
            verbose=verbose,
            batch_size=batch_size,
            max_nbytes=max_nbytes,
        )(
            delayed(func_dec)(qp.environ, qc.env, None if seed is None else seed + i, args_i)
            for i, args_i in enumerate(args_list)
        )

    if return_as == "array":
        out = np.asarray(out)
    return out


def get_shift(test_prevs: np.ndarray, train_prev: np.ndarray | float, decimals=2):
    """
    Computes the shift of an array of prevalence values for a set of test sample in
    relation to the prevalence value of the training set.

    :param test_prevs: prevalence values for the test samples
    :param train_prev: prevalence value for the training set
    :param decimals: rounding decimals for the result (default=2)
    :return: an ndarray with the shifts for each test sample, shaped as (n,1) (ndim=2)
    """
    if test_prevs.ndim == 1:
        test_prevs = test_prevs[:, np.newaxis]
    train_prevs = np.tile(train_prev, (test_prevs.shape[0], 1))
    # _shift = nae(test_prevs, train_prevs)
    _shift = qp.error.ae(test_prevs, train_prevs)
    return np.around(_shift, decimals=decimals)


def contingency_table(y, y_hat, n_classes):
    ct = np.zeros((n_classes, n_classes))
    for _c in range(n_classes):
        _idx = y == _c
        for _c1 in range(n_classes):
            ct[_c, _c1] = np.sum(y_hat[_idx] == _c1)

    return ct / y.shape[0]
