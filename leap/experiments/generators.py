import os

import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import UCI_BINARY_DATASETS, UCI_MULTICLASS_DATASETS, fetch_UCIMulticlassDataset
from quapy.method._kdey import KDEyML
from quapy.method.aggregative import ACC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.neural_network import MLPClassifier as MLP
from sklearn.svm import SVC

from leap.dataset import DatasetProvider as DP
from leap.error import vanilla_acc
from leap.models.cont_table import (
    LEAP,
    CAPContingencyTable,
    NaiveCAP,
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
    for dataset_name in [d for d in UCI_MULTICLASS_DATASETS if d not in ["wine-quality", "letter"]]:
        yield dataset_name, None if only_names else fetch_UCIMulticlassDataset(dataset_name)


def gen_product(gen1, gen2):
    for v1 in gen1():
        for v2 in gen2():
            yield v1, v2


### baselines ###
def gen_CAP_baselines(h, acc_fn, config, with_oracle=False) -> [str, CAPDirect]:
    yield "ATC", ATC(h, acc_fn, scoring_fn="maxconf")
    yield "DoC", DoC(h, acc_fn, sample_size=qp.environ["SAMPLE_SIZE"])


def gen_CAP_cont_table(h, acc_fn, config) -> [str, CAPContingencyTable]:
    yield "Naive", NaiveCAP(h, acc_fn)
    yield "LEAP", LEAP(h, acc_fn, ACC(LR()), reuse_h=True)
    yield "LEAP-plus", LEAP(h, acc_fn, KDEyML(LR()), reuse_h=True)


def gen_methods(h, V, config, with_oracle=False):
    config = "multiclass" if config is None else config

    _, acc_fn = next(gen_acc_measure())

    for name, method in gen_CAP_baselines(h, acc_fn, config, with_oracle):
        yield name, method, V
    for name, method in gen_CAP_cont_table(h, acc_fn, config):
        yield name, method, V


def get_method_names(config):
    mock_h = LR()
    _, mock_acc_fn = next(gen_acc_measure())
    return [m for m, _ in gen_CAP_baselines(mock_h, mock_acc_fn, config)] + [
        m for m, _ in gen_CAP_cont_table(mock_h, mock_acc_fn, config)
    ]


def gen_acc_measure(multiclass=False):
    yield "vanilla_accuracy", vanilla_acc


def any_missing(basedir, cls_name, dataset_name, method_name):
    for acc_name, _ in gen_acc_measure():
        if not os.path.exists(get_results_path(basedir, cls_name, acc_name, dataset_name, method_name)):
            return True
    return False
