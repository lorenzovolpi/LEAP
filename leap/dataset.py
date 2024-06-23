from quapy.data.base import LabelledCollection
from quapy.data.datasets import fetch_UCIBinaryDataset

from leap.environment import env

TRAIN_VAL_PROP = 0.5


class DatasetProvider:
    @classmethod
    def _split_train(cls, train: LabelledCollection):
        return train.split_stratified(TRAIN_VAL_PROP, random_state=0)

    @classmethod
    def uci_binary(cls, dataset_name, data_home=env["QUAPY_DATA"]):
        train, U = fetch_UCIBinaryDataset(dataset_name, data_home=data_home).train_test
        T, V = cls._split_train(train)
        return T, V, U
