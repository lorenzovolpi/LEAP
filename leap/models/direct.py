from typing import Callable

import numpy as np
from quapy.data.base import LabelledCollection
from quapy.protocol import UPP
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

from leap.models.base import ClassifierAccuracyPrediction
from leap.models.utils import get_posteriors_from_h, max_conf, neg_entropy


class CAPDirect(ClassifierAccuracyPrediction):
    def __init__(self, h: BaseEstimator, acc: Callable):
        super().__init__(h)
        self.acc = acc

    def true_acc(self, sample: LabelledCollection):
        y_pred = self.h.predict(sample.X)
        y_true = sample.y
        conf_table = confusion_matrix(y_true, y_pred=y_pred, labels=sample.classes_)
        return self.acc(conf_table)

    def switch_and_fit(self, acc_fn, data):
        self.acc = acc_fn
        return self.fit(data)


### baselines ###


class ATC(CAPDirect):
    VALID_FUNCTIONS = {"maxconf", "neg_entropy"}

    def __init__(self, h, acc_fn, scoring_fn="maxconf"):
        assert (
            scoring_fn in ATC.VALID_FUNCTIONS
        ), f"unknown scoring function, use any from {ATC.VALID_FUNCTIONS}"
        # assert acc_fn == 'vanilla_accuracy', \
        #    'use acc_fn=="vanilla_accuracy"; other metris are not yet tested in ATC'
        super().__init__(h, acc_fn)
        self.scoring_fn = scoring_fn

    def get_scores(self, P):
        if self.scoring_fn == "maxconf":
            scores = max_conf(P)
        else:
            scores = neg_entropy(P)
        return scores

    def fit(self, val: LabelledCollection):
        P = get_posteriors_from_h(self.h, val.X)
        pred_labels = np.argmax(P, axis=1)
        true_labels = val.y
        scores = self.get_scores(P)
        _, self.threshold = self.__find_ATC_threshold(
            scores=scores, labels=(pred_labels == true_labels)
        )
        return self

    def predict(self, X, oracle_prev=None):
        P = get_posteriors_from_h(self.h, X)
        scores = self.get_scores(P)
        # assert self.acc_fn == 'vanilla_accuracy', \
        #    'use acc_fn=="vanilla_accuracy"; other metris are not yet tested in ATC'
        return self.__get_ATC_acc(self.threshold, scores)

    def __find_ATC_threshold(self, scores, labels):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        sorted_idx = np.argsort(scores)

        sorted_scores = scores[sorted_idx]
        sorted_labels = labels[sorted_idx]

        fp = np.sum(labels == 0)
        fn = 0.0

        min_fp_fn = np.abs(fp - fn)
        thres = 0.0
        for i in range(len(labels)):
            if sorted_labels[i] == 0:
                fp -= 1
            else:
                fn += 1

            if np.abs(fp - fn) < min_fp_fn:
                min_fp_fn = np.abs(fp - fn)
                thres = sorted_scores[i]

        return min_fp_fn, thres

    def __get_ATC_acc(self, thres, scores):
        # code copy-pasted from https://github.com/saurabhgarg1996/ATC_code/blob/master/ATC_helper.py
        return np.mean(scores >= thres)


class DoC(CAPDirect):
    def __init__(self, h, acc, sample_size, num_samples=500, clip_vals=(0, 1)):
        self.h = h
        self.acc = acc
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.clip_vals = clip_vals

    def _get_post_stats(self, X, y):
        P = get_posteriors_from_h(self.h, X)
        mc = max_conf(P)
        pred_labels = np.argmax(P, axis=-1)
        acc = self.acc(y, pred_labels)
        return mc, acc

    def _doc(self, mc1, mc2):
        return mc2.mean() - mc1.mean()

    def train_regression(self, v2_mcs, v2_accs):
        docs = [self._doc(self.v1_mc, v2_mc_i) for v2_mc_i in v2_mcs]
        target = [self.v1_acc - v2_acc_i for v2_acc_i in v2_accs]
        docs = np.asarray(docs).reshape(-1, 1)
        target = np.asarray(target)
        lin_reg = LinearRegression()
        return lin_reg.fit(docs, target)

    def predict_regression(self, test_mc):
        docs = np.asarray([self._doc(self.v1_mc, test_mc)]).reshape(-1, 1)
        pred_acc = self.reg_model.predict(docs)
        return self.v1_acc - pred_acc

    def fit(self, val: LabelledCollection):
        v1, v2 = val.split_stratified(train_prop=0.5, random_state=0)

        self.v1_mc, self.v1_acc = self._get_post_stats(*v1.Xy)

        v2_prot = UPP(
            v2,
            sample_size=self.sample_size,
            repeats=self.num_samples,
            return_type="labelled_collection",
        )
        v2_stats = [self._get_post_stats(*sample.Xy) for sample in v2_prot()]
        v2_mcs, v2_accs = list(zip(*v2_stats))

        self.reg_model = self.train_regression(v2_mcs, v2_accs)

        return self

    def predict(self, X, oracle_prev=None):
        P = get_posteriors_from_h(self.h, X)
        mc = max_conf(P)
        acc_pred = self.predict_regression(mc)[0]
        if self.clip_vals is not None:
            acc_pred = np.clip(acc_pred, *self.clip_vals)
        return acc_pred
