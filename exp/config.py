import os
from dataclasses import dataclass

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method.aggregative import ACC, CC, KDEyML
from quapy.protocol import UPP, AbstractStochasticSeededProtocol
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

import leap
from leap.commons import contingency_table
from leap.data.datasets import fetch_UCIBinaryDataset, fetch_UCIMulticlassDataset, sort_datasets_by_size
from leap.error import vanilla_acc
from leap.models.cont_table import LEAP, OCE, PHD, NaiveCAP
from leap.models.direct import ATC, DoC
from leap.models.utils import OracleQuantifier

PROJECT = "leap"
root_dir = os.path.join(leap.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
CSV_SEP = ","

PROBLEM = "multiclass"


def split_validation(V: LabelledCollection, ratio=0.6, repeats=100):
    v_train, v_val = V.split_stratified(ratio, random_state=qp.environ["_R_SEED"])
    val_prot = UPP(v_val, repeats=repeats, return_type="labelled_collection")
    return v_train, val_prot


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

    def create_bundle(self, h: BaseEstimator):
        # generate test protocol
        self.test_prot = UPP(
            self.U,
            repeats=NUM_TEST,
            return_type="labelled_collection",
            random_state=qp.environ["_R_SEED"],
        )

        # split validation set
        self.V1, self.V2_prot = split_validation(self.V)

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


def gen_datasets(only_names=False):
    if PROBLEM == "binary":
        _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
        _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIBinaryDataset)
        for dn in _sorted_uci_names:
            dval = None if only_names else fetch_UCIBinaryDataset(dn)
            yield dn, dval
    elif PROBLEM == "multiclass":
        _uci_skip = ["isolet", "wine-quality", "letter"]
        _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
        _sorted_uci_names = sort_datasets_by_size(_uci_names, fetch_UCIMulticlassDataset)
        for dataset_name in _sorted_uci_names:
            dval = None if only_names else fetch_UCIMulticlassDataset(dataset_name)
            yield dataset_name, dval


def gen_acc_measure():
    yield "vanilla_accuracy", vanilla_acc


def gen_baselines(acc_fn):
    yield "Naive", NaiveCAP(acc_fn)
    yield "ATC-MC", ATC(acc_fn, scoring_fn="maxconf")


def gen_baselines_vp(acc_fn, D):
    yield "DoC", DoC(acc_fn, D.V2_prot, D.V2_prot_posteriors)


def gen_CAP_cont_table(h, acc_fn):
    yield "LEAP(CC)", LEAP(acc_fn, cc(), log_true_solve=True)
    yield "S-LEAP(CC)", PHD(acc_fn, cc())
    yield "O-LEAP(CC)", OCE(acc_fn, cc(), optim_method="SLSQP")
    yield "LEAP(ACC)", LEAP(acc_fn, acc(), reuse_h=h, log_true_solve=True)
    yield "LEAP(KDEy)", LEAP(acc_fn, kdey(), log_true_solve=True)
    yield "S-LEAP(KDEy)", PHD(acc_fn, kdey())
    yield "O-LEAP(KDEy)", OCE(acc_fn, kdey(), optim_method="SLSQP")


def gen_methods_with_oracle(h, acc_fn, D: DatasetBundle):
    oracle_q = OracleQuantifier([ui for ui in D.test_prot()])
    yield "LEAP(oracle)", LEAP(acc_fn, oracle_q, reuse_h=h, log_true_solve=True)
    yield "PHD(oracle)", PHD(acc_fn, oracle_q, reuse_h=h)
    yield "OCE(oracle)-SLSQP", OCE(acc_fn, oracle_q, reuse_h=h, optim_method="SLSQP")


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


def get_baseline_names():
    _, mock_acc_fn = next(gen_acc_measure())
    mock_D = DatasetBundle.mock()
    baselines_names = [m for m, _ in gen_baselines(mock_acc_fn)]
    baselines_vp_names = [m for m, _ in gen_baselines_vp(mock_acc_fn, mock_D)]
    return baselines_names[:2] + baselines_vp_names + baselines_names[2:]
