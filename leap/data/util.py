import quapy as qp
from quapy.data import LabelledCollection


def split_train(train: LabelledCollection, train_val_split: float):
    return train.split_stratified(train_prop=train_val_split, random_state=qp.environ["_R_SEED"])
