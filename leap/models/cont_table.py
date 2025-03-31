import itertools as IT
from abc import abstractmethod
from copy import deepcopy
from typing import Callable

import numpy as np
from quapy.data.base import LabelledCollection as LC
from quapy.functional import prevalence_from_labels
from quapy.method.aggregative import AggregativeQuantifier
from quapy.protocol import AbstractProtocol
from scipy import optimize
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix
from typing_extensions import override

from leap.models.base import ClassifierAccuracyPrediction


def _optim_minimize(loss: Callable, n_classes: int, method="SLSQP", bounds=None, constraints=None):
    """
    Searches for the optimal prevalence values, i.e., an `n_classes`-dimensional vector of the (`n_classes`-1)-simplex
    that yields the smallest lost. This optimization is carried out by means of a constrained search using scipy's
    SLSQP routine.

    :param loss: (callable) the function to minimize
    :param n_classes: (int) the number of classes, i.e., the dimensionality of the prevalence vector
    :param method: (str) the method used by scipy.optimize to minimize the loss; default="SLSQP"
    :return: (ndarray) the best prevalence vector found
    """
    # the initial point is set as the uniform distribution
    uniform_distribution = np.full(fill_value=1 / n_classes, shape=(n_classes,))

    # solutions are bounded to those contained in the unit-simplex
    r = optimize.minimize(loss, x0=uniform_distribution, method=method, bounds=bounds, constraints=constraints)
    return r.x


class LabelledCollection(LC):
    def empty_classes(self):
        """
        Returns a np.ndarray of empty classes (classes present in self.classes_ but with
        no positive instance). In case there is none, then an empty np.ndarray is returned

        :return: np.ndarray
        """
        idx = np.argwhere(self.counts() == 0).flatten()
        return self.classes_[idx]

    def non_empty_classes(self):
        """
        Returns a np.ndarray of non-empty classes (classes present in self.classes_ but with
        at least one positive instance). In case there is none, then an empty np.ndarray is returned

        :return: np.ndarray
        """
        idx = np.argwhere(self.counts() > 0).flatten()
        return self.classes_[idx]

    def has_empty_classes(self):
        """
        Checks whether the collection has empty classes

        :return: boolean
        """
        return len(self.empty_classes()) > 0

    def compact_classes(self):
        """
        Generates a new LabelledCollection object with no empty classes. It also returns a np.ndarray of
        indexes that correspond to the old indexes of the new self.classes_.

        :return: (LabelledCollection, np.ndarray,)
        """
        non_empty = self.non_empty_classes()
        all_classes = self.classes_
        old_pos = np.searchsorted(all_classes, non_empty)
        compact_classes = np.arange(len(old_pos))
        compact_y = np.array(self.y, copy=True)
        for necls, ccls in zip(non_empty, compact_classes):
            compact_y[self.y == necls] = ccls
        non_empty_collection = LabelledCollection(self.X, compact_y, classes=compact_classes)
        return non_empty_collection, old_pos


class CAPContingencyTable(ClassifierAccuracyPrediction):
    def __init__(self, acc_fn: Callable):
        self.acc_fn = acc_fn

    @abstractmethod
    def predict_ct(self, X, posteriors, oracle_prev=None) -> np.ndarray:
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

    def predict(self, X, posteriors):
        cont_table = self.predict_ct(X, posteriors)
        return self.acc_fn(cont_table)

    def _batch_predict_ct(self, prot: AbstractProtocol, posteriors):
        estim_cts = [self.predict_ct(Ui.X, posteriors=P) for Ui, P in IT.zip_longest(prot(), posteriors)]
        return estim_cts

    @override
    def batch_predict(self, prot: AbstractProtocol, posteriors, get_estim_cts=False) -> list[float]:
        estim_cts = self._batch_predict_ct(prot, posteriors)
        estim_accs = [self.acc_fn(ct) for ct in estim_cts]
        return (estim_accs, estim_cts) if get_estim_cts else estim_accs


class NaiveCAP(CAPContingencyTable):
    """
    The Naive CAP is a method that relies on the IID assumption, and thus uses the estimation in the validation data
    as an estimate for the test data.
    """

    def __init__(self, acc_fn: Callable):
        super().__init__(acc_fn)

    def fit(self, val: LabelledCollection, posteriors):
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = val.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=val.classes_)
        self.cont_table = self.cont_table / self.cont_table.sum()
        return self

    def predict_ct(self, test, posteriors):
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
        acc_fn: Callable,
        q_class: AggregativeQuantifier,
        reuse_h: BaseEstimator | None = None,
    ):
        CAPContingencyTable.__init__(self, acc_fn)
        self.reuse_h = reuse_h
        self.q_class = q_class

    def preprocess_data(self, data: LabelledCollection, posteriors):
        return data

    def prepare_quantifier(self):
        if self.reuse_h is not None:
            assert isinstance(self.q_class, AggregativeQuantifier), (
                f"quantifier {self.q_class} is not of type aggregative"
            )
            self.q = deepcopy(self.q_class)
            self.q.set_params(classifier=self.reuse_h)
        else:
            self.q = self.q_class

    def quant_classifier_fit_predict(self, data: LabelledCollection):
        if self.reuse_h is not None:
            return self.q.classifier_fit_predict(data, fit_classifier=False, predict_on=data)
        else:
            return self.q.classifier_fit_predict(data)

    def quant_aggregation_fit(self, classif_predictions: LabelledCollection, data: LabelledCollection):
        self.q.aggregation_fit(classif_predictions, data)

    def fit(self, data: LabelledCollection, posteriors):
        data = self.preprocess_data(data, posteriors)
        self.prepare_quantifier()
        classif_predictions = self.quant_classifier_fit_predict(data)
        self.quant_aggregation_fit(classif_predictions, data)
        return self

    def true_acc(self, sample: LabelledCollection, posteriors, acc_fn=None):
        y_pred = np.argmax(posteriors, axis=-1)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        acc_fn = self.acc if acc_fn is None else acc_fn
        return acc_fn(conf_table)


class ContTableTransferCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        acc_fn: Callable,
        q_class,
        reuse_h: BaseEstimator | None = None,
    ):
        super().__init__(acc_fn, q_class, reuse_h)

    def preprocess_data(self, data: LabelledCollection, posteriors):
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true=y_true, y_pred=y_hat, labels=data.classes_, normalize="all")
        self.train_prev = data.prevalence()
        return data

    def predict_ct(self, test, posteriors):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """
        prev_hat = self.q.quantify(test)
        adjustment = prev_hat / self.train_prev
        return self.cont_table * adjustment[:, np.newaxis]


class NsquaredEquationsCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        acc_fn: Callable,
        q_class,
        always_optimize=False,
        log_true_solve=False,
        reuse_h: BaseEstimator | None = None,
        verbose=False,
    ):
        super().__init__(acc_fn, q_class, reuse_h)
        self.verbose = verbose
        self.always_optimize = always_optimize
        self.log_true_solve = log_true_solve
        self._true_solve_log = []

    def _sout(self, *msgs, **kwargs):
        if self.verbose:
            print(*msgs, **kwargs)

    def preprocess_data(self, data: LabelledCollection, posteriors):
        self.classes_ = data.classes_
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=data.classes_)
        self.A, self.partial_b = self._construct_equations()
        return data

    def _construct_equations(self):
        # we need a n x n matrix of unknowns
        n = self.cont_table.shape[1]

        # Idx is the matrix of indexes of unknowns. For example, if we need the counts of
        # all instances belonging to class i that have been classified as belonging to 0, 1, ..., n:
        # the indexes of the corresponding unknowns are given by I[i,:]
        Idx = np.arange(n * n).reshape(n, n)

        # system of equations: Ax=b, A.shape=(n*n, n*n,), b.shape=(n*n,)
        A = np.zeros(shape=(n * n, n * n))
        b = np.zeros(shape=(n * n))

        # first equation: the sum of all unknowns is 1
        eq_no = 0
        A[eq_no, :] = 1
        b[eq_no] = 1
        eq_no += 1

        # (n-1)*(n-1) equations: the class cond ratios should be the same in training and in test due to the PPS assumptions.
        class_cond_ratios_tr = self.cont_table / self.cont_table.sum(axis=1, keepdims=True)
        for i in range(1, n):
            for j in range(1, n):
                ratio_ij = class_cond_ratios_tr[i, j]
                A[eq_no, Idx[i, :]] = -ratio_ij
                A[eq_no, Idx[i, j]] = 1 - ratio_ij
                b[eq_no] = 0
                eq_no += 1

        # n-1 equations: the sum of class-cond counts must equal the C&C prevalence prediction
        for i in range(1, n):
            A[eq_no, Idx[:, i]] = 1
            eq_no += 1

        # n-1 equations: the sum of true true class-conditional positives must equal the class prev label in test
        for i in range(1, n):
            A[eq_no, Idx[i, :]] = 1
            eq_no += 1

        return A, b

    def predict_ct(self, test, posteriors):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """

        n = self.cont_table.shape[1]

        h_label_preds = np.argmax(posteriors, axis=-1)

        cc_prev_estim = prevalence_from_labels(h_label_preds, self.classes_)
        q_prev_estim = self.q.quantify(test)

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)

        b[-2 * (n - 1) : -(n - 1)] = cc_prev_estim[1:]
        b[-(n - 1) :] = q_prev_estim[1:]

        # try the fast solution (may not be valid)
        x = np.linalg.solve(A, b)

        _true_solve = True
        n_classes = n**2
        if any(x < 0) or not np.isclose(x.sum(), 1) or self.always_optimize:
            self._sout("L", end="")
            _true_solve = False

            # try the iterative solution
            def loss(x):
                return np.linalg.norm(A @ x - b, ord=2)

            x = _optim_minimize(
                loss,
                n_classes=n_classes,
                bounds=tuple((0, 1) for _ in range(n_classes)),  # values in [0,1]
                constraints={"type": "eq", "fun": lambda x: 1 - sum(x)},  # values summing up to 1
            )
        else:
            self._sout(".", end="")

        cont_table_test = x.reshape(n, n)

        if self.log_true_solve:
            self._true_solve_log.append([_true_solve])

        return cont_table_test

    def batch_predict(self, prot: AbstractProtocol, posteriors, get_estim_cts=False):
        estim_cts = self._batch_predict_ct(prot, posteriors)
        estim_accs = [self.acc_fn(ct) for ct in estim_cts]
        if self.log_true_solve:
            _prot_logs = np.array(self._true_solve_log[-prot.total() :]).flatten().tolist()
            self._true_solve_log = self._true_solve_log[: -prot.total()] + [_prot_logs]
        return (estim_accs, estim_cts) if get_estim_cts else estim_accs


class OverConstrainedEquationsCAP(CAPContingencyTableQ):
    """ """

    def __init__(
        self,
        acc_fn: Callable,
        q_class,
        reuse_h: BaseEstimator | None = None,
        optim_method: str = "SLSQP",
        verbose=False,
    ):
        super().__init__(acc_fn, q_class, reuse_h)
        self.verbose = verbose
        self.optim_method = self._check_optim_method(optim_method)

    def _sout(self, *msgs, **kwargs):
        if self.verbose:
            print(*msgs, **kwargs)

    def _check_optim_method(self, method):
        _valid_methods = ["SLSQP", "SLSQP-c", "L-BFGS-B"]
        if method not in _valid_methods:
            raise ValueError(f"Invalid optimization method: {method}; valid methods are: {_valid_methods}")
        return method

    def preprocess_data(self, data: LabelledCollection, posteriors):
        self.classes_ = data.classes_
        y_hat = np.argmax(posteriors, axis=-1)
        y_true = data.y
        self.cont_table = confusion_matrix(y_true, y_pred=y_hat, labels=data.classes_)
        self.A, self.partial_b = self._construct_equations()
        return data

    def _construct_equations(self):
        # we need a n x n matrix of unknowns
        n = self.cont_table.shape[1]
        n_eqs = n * n + 2 * n + 1
        n_unknowns = n * n

        # Idx is the matrix of indexes of unknowns. For example, if we need the counts of
        # all instances belonging to class i that have been classified as belonging to 0, 1, ..., n:
        # the indexes of the corresponding unknowns are given by I[i,:]
        Idx = np.arange(n * n).reshape(n, n)

        # system of equations: Ax=b, A.shape=(n*n, n*n,), b.shape=(n*n,)
        A = np.zeros(shape=(n_eqs, n_unknowns))
        b = np.zeros(shape=(n_eqs))

        # first equation: the sum of all unknowns is 1
        eq_no = 0
        A[eq_no, :] = 1
        b[eq_no] = 1
        eq_no += 1

        # (n-1)*(n-1) equations: the class cond ratios should be the same in training and in test due to the PPS assumptions.
        class_cond_ratios_tr = self.cont_table / self.cont_table.sum(axis=1, keepdims=True)
        for i in range(n):
            for j in range(n):
                ratio_ij = class_cond_ratios_tr[i, j]
                A[eq_no, Idx[i, :]] = -ratio_ij
                A[eq_no, Idx[i, j]] = 1 - ratio_ij
                b[eq_no] = 0
                eq_no += 1

        # n-1 equations: the sum of class-cond counts must equal the C&C prevalence prediction
        for i in range(n):
            A[eq_no, Idx[:, i]] = 1
            eq_no += 1

        # n-1 equations: the sum of true true class-conditional positives must equal the class prev label in test
        for i in range(n):
            A[eq_no, Idx[i, :]] = 1
            eq_no += 1

        return A, b

    def predict_ct(self, test, posteriors):
        """
        :param test: test collection (ignored)
        :param oracle_prev: np.ndarray with the class prevalence of the test set as estimated by
            an oracle. This is meant to test the effect of the errors in CAP that are explained by
            the errors in quantification performance
        :return: a confusion matrix in the return format of `sklearn.metrics.confusion_matrix`
        """

        n = self.cont_table.shape[1]

        h_label_preds = np.argmax(posteriors, axis=-1)

        cc_prev_estim = prevalence_from_labels(h_label_preds, self.classes_)
        q_prev_estim = self.q.quantify(test)

        A = self.A
        b = self.partial_b

        # b is partially filled; we finish the vector by plugin in the classify and count
        # prevalence estimates (n-1 values only), and the quantification estimates (n-1 values only)

        b[-2 * n : -n] = cc_prev_estim
        b[-n:] = q_prev_estim

        def loss(x):
            return np.linalg.norm(A @ x - b, ord=2)

        n_classes = n**2
        if self.optim_method == "SLSQP":
            x = _optim_minimize(
                loss,
                n_classes=n_classes,
                method=self.optim_method,
                bounds=tuple((0, 1) for _ in range(n_classes)),  # values in [0,1]
                constraints={"type": "eq", "fun": lambda x: 1 - sum(x)},  # values summing up to 1
            )
        elif self.optim_method == "SLSQP-c":
            self.optim_method = "SLSQP"
            Idx = np.arange(n * n).reshape(n, n)
            _constraints = [{"type": "eq", "fun": lambda x: 1 - sum(x)}]
            for i in range(n):
                _mask = np.zeros(n_classes)
                _mask[Idx[:, i]] = 1
                _constr = {
                    "type": "eq",
                    "fun": lambda x: (x * _mask).sum() - cc_prev_estim[i],
                }
                _constraints.append(_constr)
            x = _optim_minimize(
                loss,
                n_classes=n_classes,
                method=self.optim_method,
                bounds=tuple((0, 1) for _ in range(n_classes)),  # values in [0,1]
                constraints=_constraints,
            )
        elif self.optim_method == "L-BFGS-B":
            x = _optim_minimize(
                loss,
                n_classes=n_classes,
                method=self.optim_method,
                bounds=tuple((0, 1) for _ in range(n_classes)),  # values in [0,1]
            )
            x = x / x.sum()

        cont_table_test = x.reshape(n, n)
        return cont_table_test


LEAP = NsquaredEquationsCAP
PHD = ContTableTransferCAP
OCE = OverConstrainedEquationsCAP
