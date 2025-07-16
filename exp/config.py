from dataclasses import dataclass

import numpy as np
import pandas as pd
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import ACC, CC, KDEyML
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import exp.env as env
from leap.commons import contingency_table
from leap.data.datasets import (
    fetch_UCIBinaryDataset,
    fetch_UCIMulticlassDataset,
    sort_datasets_by_size,
)
from leap.error import f1, f1_macro, vanilla_acc
from leap.models.cont_table import CBPE, LEAP, O_LEAP, S_LEAP, NaiveCAP
from leap.models.direct import ATC, COT, Q_COT, DispersionScore, DoC, NuclearNorm
from leap.models.utils import OracleQuantifier

_toggle = {
    "mlp": True,
    "same_h": True,
    "vanilla": True,
    "f1": True,
    "cc": True,
    "acc": True,
    "slsqp": False,
    "oracle": False,
}


def split_validation(V: LabelledCollection, ratio=0.6, repeats=100, sample_size=None):
    v_train, v_val = V.split_stratified(ratio, random_state=qp.environ["_R_SEED"])
    val_prot = UPP(v_val, repeats=repeats, sample_size=sample_size, return_type="labelled_collection")
    return v_train, val_prot


@dataclass
class EXP:
    code: int
    cls_name: str
    dataset_name: str
    acc_name: str
    method_name: str
    df: pd.DataFrame = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


@dataclass
class DatasetBundle:
    L_prevalence: np.ndarray
    V: LabelledCollection
    U: LabelledCollection
    V1: LabelledCollection = None
    V2_prot: AbstractStochasticSeededProtocol = None
    test_prot: AbstractStochasticSeededProtocol = None
    V_posteriors: np.ndarray = None
    V1_posteriors: np.ndarray = None
    V2_prot_posteriors: np.ndarray = None
    test_prot_posteriors: np.ndarray = None
    test_prot_y_hat: np.ndarray = None
    test_prot_true_cts: np.ndarray = None

    def get_test_prot(self, sample_size=None):
        return UPP(
            self.U,
            repeats=env.NUM_TEST,
            sample_size=sample_size,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

    def create_bundle(self, h: BaseEstimator, sample_size=None):
        # generate test protocol
        self.test_prot = self.get_test_prot(sample_size=sample_size)
        # split validation set
        self.V1, self.V2_prot = split_validation(self.V, sample_size=sample_size)

        # precomumpute model posteriors for validation sets
        self.V_posteriors = h.predict_proba(self.V.X)
        self.V1_posteriors = h.predict_proba(self.V1.X)
        self.V2_prot_posteriors = []
        for sample in self.V2_prot():
            self.V2_prot_posteriors.append(h.predict_proba(sample.X))

        # precomumpute model posteriors for test samples
        self.test_prot_posteriors, self.test_prot_y_hat, self.test_prot_true_cts = [], [], []
        for sample in self.test_prot():
            P = h.predict_proba(sample.X)
            self.test_prot_posteriors.append(P)
            y_hat = np.argmax(P, axis=-1)
            self.test_prot_y_hat.append(y_hat)
            self.test_prot_true_cts.append(contingency_table(sample.y, y_hat, sample.n_classes))

        return self

    @classmethod
    def mock(cls):
        return DatasetBundle(None, None, None, test_prot=lambda: [])


def cc():
    return CC(MLP())


def acc():
    return ACC(MLP())


def kdey():
    return KDEyML(MLP())


def gen_classifiers():
    yield "LR", LogisticRegression()
    yield "kNN", KNN(n_neighbors=10)
    yield "SVM", SVC(kernel="rbf", probability=True)
    yield "MLP", MLP()
    yield "RFC", RFC()


def gen_datasets(only_names=False):
    if env.PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif env.PROBLEM == "multiclass":
        # _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_skip = []  # ["wine-quality"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        for dataset_name in _sorted_uci_names:
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


def gen_acc_measure():
    if _toggle["vanilla"]:
        yield "vanilla_accuracy", vanilla_acc
    if _toggle["f1"]:
        if env.PROBLEM == "binary":
            yield "f1", f1
        elif env.PROBLEM == "multiclass":
            yield "macro_f1", f1_macro


def gen_baselines(acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")
    yield "DS", DispersionScore(acc_fn)
    yield "COT", COT(acc_fn)
    yield "CBPE", CBPE(acc_fn)
    yield "NN", NuclearNorm(acc_fn)
    yield "Q-COT", Q_COT(acc_fn, kdey())


def gen_baselines_vp(acc_fn, D):
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors)


# NOTE: the reason why mlp beats lr could be two-fold:
# (i) mlp uses a hidden layer of 100 ReLU which has proven to be more effective in practical applications than sigmoid function;
# (ii) while both mlp and lr use the same loss function (cross-entropy), lr corrects its predictions using a penalty function
# which is based on L2. This could affect negatively the performance of the KDEyML quantifier which uses the KL divergence as
# a loss function, which is strictly correlated to cross-entropy. The lr L2 regularization could possibly "ruin" the pure
# cross-entropy minimization towards which also KDEyML works.
def gen_CAP_cont_table(h, acc_fn):
    if _toggle["same_h"]:
        if _toggle["cc"]:
            yield "LEAP(CC)", LEAP(acc_fn, cc(), reuse_h=h, log_true_solve=True)
            yield "S-LEAP(CC)", S_LEAP(acc_fn, cc(), reuse_h=h)
            yield "O-LEAP(CC)", O_LEAP(acc_fn, cc(), reuse_h=h)
        if _toggle["acc"]:
            yield "LEAP(ACC)", LEAP(acc_fn, acc(), reuse_h=h, log_true_solve=True)
        yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True)
        yield "S-LEAP(KDEy)", S_LEAP(acc_fn, kdey(), reuse_h=h)
        yield "O-LEAP(KDEy)", O_LEAP(acc_fn, kdey(), reuse_h=h)

    if _toggle["mlp"]:
        if _toggle["cc"]:
            yield "LEAP(CC-MLP)", LEAP(acc_fn, cc(), log_true_solve=True)
            yield "S-LEAP(CC-MLP)", S_LEAP(acc_fn, cc())
            yield "O-LEAP(CC-MLP)", O_LEAP(acc_fn, cc())
        if _toggle["acc"]:
            yield "LEAP(ACC-MLP)", LEAP(acc_fn, acc(), log_true_solve=True)
        yield "LEAP(KDEy-MLP)", LEAP(acc_fn, kdey(), log_true_solve=True)
        yield "S-LEAP(KDEy-MLP)", S_LEAP(acc_fn, kdey())
        yield "O-LEAP(KDEy-MLP)", O_LEAP(acc_fn, kdey())

    if _toggle["slsqp"]:
        if _toggle["same_h"]:
            if _toggle["cc"]:
                yield (
                    "LEAP(CC)-SLSQP",
                    LEAP(acc_fn, cc(), reuse_h=h, log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
                yield "O-LEAP(CC)-SLSQP", O_LEAP(acc_fn, cc(), reuse_h=h, optim_method="SLSQP", sparse_matrix=False)
            if _toggle["acc"]:
                yield (
                    "LEAP(ACC)-SLSQP",
                    LEAP(acc_fn, acc(), reuse_h=h, log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
            yield (
                "LEAP(KDEy)-SLSQP",
                LEAP(acc_fn, kdey(), reuse_h=h, log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
            )
            yield "O-LEAP(KDEy)-SLSQP", O_LEAP(acc_fn, kdey(), reuse_h=h, optim_method="SLSQP", sparse_matrix=False)

        if _toggle["mlp"]:
            if _toggle["cc"]:
                yield (
                    "LEAP(CC-MLP)-SLSQP",
                    LEAP(acc_fn, cc(), log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
                yield "O-LEAP(CC-MLP)-SLSQP", O_LEAP(acc_fn, cc(), optim_method="SLSQP", sparse_matrix=False)
            if _toggle["acc"]:
                yield (
                    "LEAP(ACC-MLP)-SLSQP",
                    LEAP(acc_fn, acc(), log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
                )
            yield (
                "LEAP(KDEy-MLP)-SLSQP",
                LEAP(acc_fn, kdey(), log_true_solve=True, optim_method="SLSQP", sparse_matrix=False),
            )
            yield "O-LEAP(KDEy-MLP)-SLSQP", O_LEAP(acc_fn, kdey(), optim_method="SLSQP", sparse_matrix=False)


def gen_methods_with_oracle(h, acc_fn, D: DatasetBundle):
    if _toggle["oracle"]:
        oracle_q = OracleQuantifier([ui for ui in D.test_prot()])
        yield (
            "LEAP(oracle)",
            LEAP(acc_fn, oracle_q, reuse_h=h, log_true_solve=True),
        )
        yield "S-LEAP(oracle)", S_LEAP(acc_fn, oracle_q, reuse_h=h)
        yield "O-LEAP(oracle)", O_LEAP(acc_fn, oracle_q, reuse_h=h)
    else:
        return
        yield


def gen_methods(h, D):
    _, acc_fn = next(gen_acc_measure())
    for name, method in gen_baselines(acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_baselines_vp(acc_fn, D):
        yield name, method, D.V1, D.V1_posteriors
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, D.V, D.V_posteriors
    for name, method in gen_methods_with_oracle(h, acc_fn, D):
        yield name, method, D.V, D.V_posteriors


def get_classifier_names():
    return [name for name, _ in gen_classifiers()]


def get_dataset_names():
    return [name for name, _ in gen_datasets(only_names=True)]


def get_acc_names():
    return [acc_name for acc_name, _ in gen_acc_measure()]


def get_method_names(with_oracle=True):
    mock_h = LogisticRegression()
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()

    baselines = [m for m, _ in gen_baselines(mock_acc_fn)] + [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_D)]
    CAP_ct = [m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)]

    names = baselines + CAP_ct

    if with_oracle:
        names += [m for m, _ in gen_methods_with_oracle(mock_h, mock_acc_fn, mock_D)]

    return names


def is_excluded(classifier, dataset, method, acc):
    return False


def get_baseline_names():
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()
    baselines_names = [m for m, _ in gen_baselines(mock_acc_fn)]
    baselines_vp_names = [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_D)]
    return baselines_names[:2] + baselines_vp_names + baselines_names[2:]
