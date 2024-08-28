import os

import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS
from quapy.method._kdey import KDEyML
from quapy.method.aggregative import ACC, CC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

from leap.dataset import DatasetProvider as DP
from leap.error import vanilla_acc
from leap.models.base import CAP
from leap.models.cont_table import (
    LEAP,
    CAPContingencyTable,
    NaiveCAP,
    NaiveRescalingCAP,
)
from leap.models.direct import ATC, CAPDirect, DoC
from leap.utils.commons import get_results_path


def gen_classifiers():
    yield "LR", LR()
    yield "KNN", KNN(n_neighbors=10)
    yield "SVM", SVC(probability=True)
    yield "MLP", MLP(hidden_layer_sizes=(100, 15), random_state=0)


def gen_bin_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    _uci_skip = ["acute.a", "acute.b", "balance.2", "iris.1"]
    _uci_names = [d for d in UCI_BINARY_DATASETS if d not in _uci_skip]
    for dn in _uci_names:
        dval = None if only_names else DP.uci_binary(dn)
        yield dn, dval


def gen_multi_datasets(
    only_names=False,
) -> [str, [LabelledCollection, LabelledCollection, LabelledCollection]]:
    # yields the UCI multiclass datasets
    _uci_skip = ["wine-quality", "letter", "isolet"]
    _uci_names = [d for d in UCI_MULTICLASS_DATASETS if d not in _uci_skip]
    for dn in _uci_names:
        dval = None if only_names else DP.uci_multiclass(dn)
        yield dn, dval


def gen_product(gen1, gen2):
    for v1 in gen1():
        for v2 in gen2():
            yield v1, v2


### baselines ###
def gen_CAP_baselines(h, acc_fn) -> [str, CAPDirect]:
    yield "ATC", ATC(h, acc_fn, scoring_fn="maxconf")
    yield "DoC", DoC(h, acc_fn, sample_size=qp.environ["SAMPLE_SIZE"])


def gen_CAP_cont_table(h, acc_fn) -> [str, CAPContingencyTable]:
    yield "Naive", NaiveCAP(h, acc_fn)
    yield "LEAPcc", LEAP(h, acc_fn, CC(LR()), reuse_h=True)
    yield "LEAP", LEAP(h, acc_fn, ACC(LR()), reuse_h=True)
    yield "LEAP-plus", LEAP(h, acc_fn, KDEyML(LR()), reuse_h=True)
    yield "NaiveRescaling", NaiveRescalingCAP(h, acc_fn, ACC(LR()), reuse_h=True)
    yield "NaiveRescaling-plus", NaiveRescalingCAP(h, acc_fn, KDEyML(LR()), reuse_h=True)


def gen_CAP_CT_with_oracle(h, acc_fn) -> [str, CAPContingencyTable]:
    yield "LEAP-oracle", LEAP(h, acc_fn, ACC(LR()), reuse_h=True)


def gen_methods(h) -> [str, CAP, bool]:
    """
    A generator to create all methods for the current experiment

    :param h: the classifier used to create the method instances
    :return: tuples comprised of name of the method, instance of the method and flag indicating
             weather the method expects an oracle
    """
    _, acc_fn = next(gen_acc_measure())

    for name, method in gen_CAP_baselines(h, acc_fn):
        yield name, method, False
    for name, method in gen_CAP_cont_table(h, acc_fn):
        yield name, method, False
    for name, method in gen_CAP_CT_with_oracle(h, acc_fn):
        yield name, method, True


def get_method_names():
    mock_h = LR()
    _, mock_acc_fn = next(gen_acc_measure())
    return [m for m, _ in gen_CAP_baselines(mock_h, mock_acc_fn)] + [
        m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn)
    ]


def gen_acc_measure(multiclass=False):
    yield "vanilla_accuracy", vanilla_acc


def any_missing(basedir, cls_name, dataset_name, method_name):
    for acc_name, _ in gen_acc_measure():
        if not os.path.exists(get_results_path(basedir, cls_name, acc_name, dataset_name, method_name)):
            return True
    return False
