import quapy as qp
from quapy.data.base import LabelledCollection
from quapy.data.datasets import fetch_UCIBinaryDataset as UCIBin
from quapy.data.datasets import fetch_UCIMulticlassDataset as UCIMulti

from leap.environment import env


def split_train(train: LabelledCollection, train_val_split: float):
    return train.split_stratified(train_prop=train_val_split, random_state=qp.environ["_R_SEED"])


def fetch_UCIBinaryDataset(dataset_name, data_home=env["QUAPY_DATA"], train_val_split=0.5):
    train, U = UCIBin(dataset_name, data_home=data_home).train_test
    T, V = split_train(train, train_val_split)
    return T, V, U


def fetch_UCIMulticlassDataset(dataset_name, data_home=env["QUAPY_DATA"], train_val_split=0.5):
    train, U = UCIMulti(dataset_name, data_home=data_home).train_test
    T, V = split_train(train, train_val_split)
    return T, V, U


def sort_datasets_by_size(dataset_names, dataset_fun, descending=True):
    def get_dataset_size(name):
        L, V, U = dataset_fun(name)
        return len(L) + len(V) + len(U)

    datasets = [(d, get_dataset_size(d)) for d in dataset_names]
    datasets.sort(key=(lambda d: d[1]), reverse=descending)
    return [d for (d, _) in datasets]
