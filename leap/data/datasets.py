import numpy as np
from quapy.data.base import LabelledCollection
from quapy.data.datasets import fetch_UCIBinaryDataset as UCIBin
from quapy.data.datasets import fetch_UCIMulticlassDataset as UCIMulti
from sklearn.datasets import fetch_rcv1

from leap.data.util import get_rcv1_class_info, split_train
from leap.environment import env


def fetch_RCV1WholeDataset(train_val_split=0.5, min_docs=5):
    """Retrieve the whole multiclass dataset extracted from the RCV1 taxonomy.
    For each class, labels from the upper hierarchy are removed, so that each document
    is labelled with the most specific class of competence.
    Other multi-label datapoints are removed from the dataset to keep it single-label.

    :param train_val_split: the propoertion to use for splitting the source set
    in training and validation.
    :return: a tuple with training, validation and test sets.
    """

    def find_parents(c, rev_tree, sorted_classes):
        if c not in rev_tree:
            return []

        try:
            parent = rev_tree[c]
            parent_idx = sorted_classes.index(parent)
            return [parent_idx] + find_parents(parent, rev_tree, sorted_classes)
        except ValueError:
            return find_parents(parent, rev_tree, sorted_classes)

    def filter_multi_label(dset, rev_tree, sorted_classes):
        labels = dset.target.toarray()
        for i, c in enumerate(sorted_classes):
            idx_i = labels[:, i] == 1
            parents = np.asarray(find_parents(c, rev_tree, sorted_classes))
            if len(parents) == 0:
                continue
            labels[idx_i, parents[:, np.newaxis]] = 0

        valid_data = labels.sum(axis=1) == 1
        return LabelledCollection(
            dset.data[valid_data, :],
            np.argmax(labels[valid_data, :], axis=1),
        )

    def filter_min_docs(train: LabelledCollection, U: LabelledCollection):
        all_classes_ = np.sort(np.unique(np.hstack([train.classes_, U.classes_])))
        _bins = np.hstack([all_classes_, [all_classes_.max() + 1]])

        train_valid_classes = np.histogram(train.y, bins=_bins)[0] >= min_docs
        U_valid_classes = np.histogram(U.y, bins=_bins)[0] >= min_docs
        valid_classes = all_classes_[train_valid_classes & U_valid_classes]

        # compute the indexes for the valid documents in train and test given the valid classes
        train_idx = np.any(train.y == valid_classes[:, np.newaxis], axis=0)
        U_idx = np.any(U.y == valid_classes[:, np.newaxis], axis=0)

        train_X = train.X[train_idx, :]
        train_y = train.y[train_idx]
        U_X = U.X[U_idx, :]
        U_y = U.y[U_idx]

        compr_classes = np.zeros(valid_classes.max() + 1)
        compr_classes[valid_classes] = np.arange(len(valid_classes))

        for c in valid_classes:
            train_y[train_y == c] = compr_classes[c]
            U_y[U_y == c] = compr_classes[c]

        return LabelledCollection(train_X, train_y, classes=np.arange(len(valid_classes))), LabelledCollection(
            U_X, U_y, classes=np.arange(len(valid_classes))
        )

    _, tree, _ = get_rcv1_class_info()
    # reverse tree
    rev_tree = {}
    for c, scs in tree.items():
        for sc in scs:
            rev_tree[str(sc)] = c

    og_training = fetch_rcv1(subset="train", data_home=env["SKLEARN_DATA"])
    og_test = fetch_rcv1(subset="test", data_home=env["SKLEARN_DATA"])

    sorted_classes = og_training.target_names.tolist()

    train = filter_multi_label(og_training, rev_tree, sorted_classes)
    U = filter_multi_label(og_test, rev_tree, sorted_classes)

    train, U = filter_min_docs(train, U)

    T, V = split_train(train, train_val_split)
    return T, V, U
    # return len(train), len(U), train.n_classes


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
