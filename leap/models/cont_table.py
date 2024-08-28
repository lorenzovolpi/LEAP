from abc import abstractmethod
from copy import deepcopy
from typing import Callable

import numpy as np
import quapy.functional as F
from quapy.data.base import LabelledCollection
from quapy.method.aggregative import AggregativeQuantifier
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from leap.models.base import ClassifierAccuracyPrediction


class CAPContingencyTable(ClassifierAccuracyPrediction):
    def __init__(self, h: BaseEstimator, acc_fn: Callable):
        super().__init__(h)
        self.acc_fn = acc_fn

    @abstractmethod
    def predict_ct(self, X, oracle_prev=None) -> np.ndarray:
        """
        Predicts the contingency table for the test data

        :param X: test data
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a contingency table
        """
        ...

    def switch(self, acc_fn):
        self.acc_fn = acc_fn
        return self

    def predict(self, data: LabelledCollection, oracle_prev=None):
        cont_table = self.predict_ct(data, oracle_prev)
        return self.acc_fn(cont_table)


class NaiveCAP(CAPContingencyTable):
    """
    The Naive CAP is a method that relies on the IID assumption, and thus uses the estimation in the validation data
    as an estimate for the test data.
    """

    def __init__(self, h: BaseEstimator, acc_fn: Callable):
        super().__init__(h, acc_fn)

    def fit(self, val: LabelledCollection):
        y_hat = self.h.predict(val.X)
        y_true = val.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=val.classes_)
        return self

    def predict_ct(self, test, oracle_prev=None):
        """
        This method disregards the test set, under the assumption that it is IID wrt the training. This meaning that
        the confusion matrix for the test data should coincide with the one computed for training (using any cross
        validation strategy).

        :param test: test collection (ignored)
        :param oracle_prev: ignored
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """
        return self.cont_table


class CAPContingencyTableQ(CAPContingencyTable, BaseEstimator):
    def __init__(
        self,
        h: BaseEstimator,
        acc_fn: Callable,
        q_class: AggregativeQuantifier,
        reuse_h=False,
    ):
        CAPContingencyTable.__init__(self, h, acc_fn)
        self.reuse_h = reuse_h
        self.q_class = q_class

    def preprocess_data(self, data: LabelledCollection):
        return data

    def prepare_quantifier(self):
        if self.reuse_h:
            assert isinstance(
                self.q_class, AggregativeQuantifier
            ), f"quantifier {self.q_class} is not of type aggregative"
            self.q = deepcopy(self.q_class)
            self.q.set_params(classifier=self.h)
        else:
            self.q = self.q_class

    def quant_classifier_fit_predict(self, data: LabelledCollection):
        if self.reuse_h:
            return self.q.classifier_fit_predict(data, fit_classifier=False, predict_on=data)
        else:
            return self.q.classifier_fit_predict(data)

    def quant_aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.q.aggregation_fit(classif_predictions, data)

    def fit(self, data: LabelledCollection):
        data = self.preprocess_data(data)
        self.prepare_quantifier()
        classif_predictions = self.quant_classifier_fit_predict(data)
        self.quant_aggregation_fit(classif_predictions, data)
        return self


class ContTableTransferCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        h: BaseEstimator,
        acc_fn: Callable,
        q_class,
        reuse_h=False,
    ):
        super().__init__(h, acc_fn, q_class, reuse_h)

    def preprocess_data(self, data: LabelledCollection, posteriors):
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true=y_true, y_pred=y_hat, labels=data.classes_, normalize="all")
        self.train_prev = data.prevalence()
        return data

    def predict_ct(self, test, posteriors, oracle_prev=None):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """
        if oracle_prev is None:
            prev_hat = self.q.quantify(test)
        else:
            prev_hat = oracle_prev
        adjustment = prev_hat / self.train_prev
        return self.cont_table * adjustment[:, np.newaxis]


class NsquaredEquationsCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        h: BaseEstimator,
        acc_fn: Callable,
        q_class,
        always_optimize=False,
        reuse_h=False,
        verbose=False,
    ):
        super().__init__(h, acc_fn, q_class, reuse_h)
        self.verbose = verbose
        self.always_optimize = always_optimize

    def _sout(self, *msgs, **kwargs):
        if self.verbose:
            print(*msgs, **kwargs)

    def preprocess_data(self, data: LabelledCollection):
        y_hat = self.h.predict(data.X)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=data.classes_)
        self.A, self.partial_b = self._construct_equations()
        return data

    def _construct_equations(self):
        # we need a n x n matrix of unknowns
        n = self.cont_table.shape[1]

        # I is the matrix of indexes of unknowns. For example, if we need the counts of
        # all instances belonging to class i that have been classified as belonging to 0, 1, ..., n:
        # the indexes of the corresponding unknowns are given by I[i,:]
        I = np.arange(n * n).reshape(n, n)

        # system of equations: Ax=b, A.shape=(n*n, n*n,), b.shape=(n*n,)
        A = np.zeros(shape=(n * n, n * n))
        b = np.zeros(shape=(n * n))

        # first equation: the sum of all unknowns is 1
        eq_no = 0
        A[eq_no, :] = 1
        b[eq_no] = 1
        eq_no += 1

        # (n-1)*(n-1) equations: the class cond ratios should be the same in training and in test due to the
        # PPS assumptions. Example in three classes, a ratio: a/(a+b+c) [test] = ar [a ratio in training]
        # a / (a + b + c) = ar
        # a = (a + b + c) * ar
        # a = a ar + b ar + c ar
        # a - a ar - b ar - c ar = 0
        # a (1-ar) + b (-ar)  + c (-ar) = 0
        class_cond_ratios_tr = self.cont_table / self.cont_table.sum(axis=1, keepdims=True)
        for i in range(1, n):
            for j in range(1, n):
                ratio_ij = class_cond_ratios_tr[i, j]
                A[eq_no, I[i, :]] = -ratio_ij
                A[eq_no, I[i, j]] = 1 - ratio_ij
                b[eq_no] = 0
                eq_no += 1

        # n-1 equations: the sum of class-cond counts must equal the C&C prevalence prediction
        for i in range(1, n):
            A[eq_no, I[:, i]] = 1
            # b[eq_no] = cc_prev_estim[i]
            eq_no += 1

        # n-1 equations: the sum of true true class-conditional positives must equal the class prev label in test
        for i in range(1, n):
            A[eq_no, I[i, :]] = 1
            # b[eq_no] = q_prev_estim[i]
            eq_no += 1

        return A, b

    def predict_ct(self, test, oracle_prev=None, return_true_solve=False):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """

        n = self.cont_table.shape[1]

        h_label_preds = self.h.predict(test)
        cc_prev_estim = F.prevalence_from_labels(h_label_preds, self.h.classes_)
        if oracle_prev is None:
            q_prev_estim = self.q.quantify(test)
        else:
            q_prev_estim = oracle_prev

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)

        b[-2 * (n - 1) : -(n - 1)] = cc_prev_estim[1:]
        b[-(n - 1) :] = q_prev_estim[1:]

        # try the fast solution (may not be valid)
        x = np.linalg.solve(A, b)

        _true_solve = True
        if any(x < 0) or not np.isclose(x.sum(), 1) or self.always_optimize:
            self._sout("L", end="")
            _true_solve = False

            # try the iterative solution
            def loss(x):
                return np.linalg.norm(A @ x - b, ord=2)

            x = F.optim_minimize(loss, n_classes=n**2)

        else:
            self._sout(".", end="")

        cont_table_test = x.reshape(n, n)
        if return_true_solve:
            return cont_table_test, _true_solve
        else:
            return cont_table_test


LEAP = NsquaredEquationsCAP
