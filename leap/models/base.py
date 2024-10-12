import itertools as IT
from abc import ABC, abstractmethod
from typing import List

from quapy.data.base import LabelledCollection
from quapy.protocol import AbstractProtocol
from sklearn.base import BaseEstimator


class ClassifierAccuracyPrediction(ABC):
    def __init__(self, h: BaseEstimator):
        self.h = h

    @abstractmethod
    def fit(self, val: LabelledCollection):
        """
        Trains a CAP method.

        :param val: training data
        :return: self
        """
        ...

    @abstractmethod
    def predict(self, X, oracle_prev=None) -> float:
        """
        Predicts directly the accuracy using the accuracy function

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: float
        """
        ...

    def batch_predict(self, prot: AbstractProtocol, oracle_prevs=None) -> list[float]:
        if oracle_prevs is None:
            estim_accs = [self.predict(Ui.X) for Ui in prot()]
            return estim_accs
        else:
            assert isinstance(oracle_prevs, List), "Invalid oracles"
            estim_accs = [self.predict(Ui.X, oracle_prev=op) for Ui, op in IT.zip_longest(prot(), oracle_prevs)]
            return estim_accs


CAP = ClassifierAccuracyPrediction
