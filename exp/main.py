import itertools as IT
import os
from traceback import print_exception

import numpy as np
import pandas as pd
import quacc as qc
import quapy as qp
from sklearn.base import clone as skl_clone

from exp.config import (
    PROBLEM,
    PROJECT,
    DatasetBundle,
    gen_acc_measure,
    gen_classifiers,
    gen_datasets,
    gen_methods,
    get_method_names,
    root_dir,
)
from exp.util import (
    fit_or_switch,
    gen_model_dataset,
    get_ct_predictions,
    get_logger,
    get_plain_prev,
    timestamp,
)
from leap.commons import get_shift, parallel, true_acc
from leap.models.cont_table import LEAP

log = get_logger(id=PROJECT)

qp.environ["SAMPLE_SIZE"] = 100


def local_path(dataset_name, cls_name, method_name, acc_name):
    parent_dir = os.path.join(root_dir, PROBLEM, cls_name, acc_name, dataset_name)
    os.makedirs(parent_dir, exist_ok=True)
    return os.path.join(parent_dir, f"{method_name}.json")


def is_excluded(classifier, dataset, method, acc):
    return False


def get_extra_from_method(df, method):
    if isinstance(method, LEAP):
        df["true_solve"] = method._true_solve_log[-1]


def all_exist_pre_check(dataset_name, cls_name):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        if is_excluded(cls_name, dataset_name, method, acc):
            continue
        path = local_path(dataset_name, cls_name, method, acc)
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def gen_method_df(df_len, **data):
    data = data | {k: [v] * df_len for k, v in data.items() if not isinstance(v, list)}
    return pd.DataFrame.from_dict(data, orient="columns")


class EXP:
    def __init__(self, code, err=None):
        self.code = code
        self.err = err

    @classmethod
    def SUCCESS(cls):
        return EXP(200)

    @classmethod
    def EXISTS(cls):
        return EXP(300)

    @classmethod
    def ERROR(cls, e):
        return EXP(400, err=e)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


def exp_protocol(args):
    cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors = args
    results = []

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    t_train = None
    for acc_name, acc_fn in gen_acc_measure():
        if is_excluded(cls_name, dataset_name, method_name, acc_name):
            continue
        path = local_path(dataset_name, cls_name, method_name, acc_name)
        if os.path.exists(path):
            results.append((cls_name, dataset_name, acc_name, method_name, None, None, None, EXP.EXISTS()))
            continue

        try:
            method, _t_train = fit_or_switch(method, val, val_posteriors, acc_fn, t_train is not None)
            t_train = t_train if _t_train is None else _t_train

            test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
            estim_accs, estim_cts, t_test_ave = get_ct_predictions(method, D.test_prot, D.test_prot_posteriors)
            if estim_cts is None:
                estim_cts = [None] * len(estim_accs)
            else:
                estim_cts = [ct.tolist() for ct in estim_cts]
        except Exception as e:
            print_exception(e)
            results.append((cls_name, dataset_name, acc_name, method_name, None, None, None, EXP.ERROR(e)))
            continue

        ae = qc.error.ae(np.array(true_accs[acc_name]), np.array(estim_accs)).tolist()

        df_len = len(estim_accs)
        method_df = gen_method_df(
            df_len,
            shifts=test_shift,
            true_accs=true_accs[acc_name],
            estim_accs=estim_accs,
            acc_err=ae,
            estim_cts=estim_cts,
            true_cts=D.test_prot_true_cts,
            classifier=cls_name,
            method=method_name,
            dataset=dataset_name,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )
        get_extra_from_method(method_df, method)

        results.append((cls_name, dataset_name, acc_name, method_name, method_df, t_train, t_test_ave, EXP.SUCCESS()))

    return results


def experiments():
    cls_dataset = []
    for (cls_name, orig_h), (dataset_name, (L, V, U)) in gen_model_dataset(gen_classifiers, gen_datasets):
        # check if all results for current combination already exist
        # if so, skip the combination
        if all_exist_pre_check(dataset_name, cls_name):
            log.info(f"All results for {cls_name} over {dataset_name} exist, skipping")
        else:
            # clone model from the original one
            h = skl_clone(orig_h)
            # fit model
            h.fit(*L.Xy)
            log.info(f"Trained {cls_name} over {dataset_name}")
            # create dataset bundle
            D = DatasetBundle(L.prevalence(), V, U).create_bundle(h)
            # compute true accs for h on dataset
            true_accs = {}
            for acc_name, acc_fn in gen_acc_measure():
                true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in D.test_prot()]
            # store h-dataset combination
            cls_dataset.append((cls_name, dataset_name, h, D, true_accs))

    exp_prot_args_list = []
    for cls_name, dataset_name, h, D, true_accs in cls_dataset:
        for method_name, method, val, val_posteriors in gen_methods(h, D):
            exp_prot_args_list.append(
                (cls_name, dataset_name, h, D, true_accs, method_name, method, val, val_posteriors)
            )

    results_gen = parallel(
        func=exp_protocol,
        args_list=exp_prot_args_list,
        n_jobs=qc.env["N_JOBS"],
        return_as="generator_unordered",
    )

    for res in results_gen:
        for cls_name, dataset_name, acc_name, method_name, method_df, t_train, t_test_ave, r in res:
            if r.ok:
                path = local_path(dataset_name, cls_name, method_name, acc_name)
                method_df.to_json(path)
                log.info(
                    f"[{cls_name}@{dataset_name}] {method_name} on {acc_name} done [{timestamp(t_train, t_test_ave)}]"
                )
            elif r.old:
                log.info(f"[{cls_name}@{dataset_name}] {method_name} on {acc_name} exists, skipping")
            elif r.error:
                log.warning(f"[{cls_name}@{dataset_name}] {method_name}: {acc_name} gave error '{r.err}' - skipping")


# TODO: parallel support for transformers
#
# def transofrmers():
#     for (cls_name, h), (dataset_name, (V, U), L_prev) in gen_transformer_model_dataset():
#         # compute and set the SAMPLE_SIZE for each dataset
#         qp.environ["SAMPLE_SIZE"] = 1000
#
#         # check if all results for current combination already exist
#         # if so, skip the combination
#         if all_exist_pre_check(dataset_name, cls_name):
#             log.info(f"All results for {cls_name} over {dataset_name} exist, skipping")
#             continue
#
#         log.info(f"Computing {cls_name} over {dataset_name}")
#         D = DatasetBundle(L_prev, V, U).create_bundle(h)
#         exp_protocol(cls_name, dataset_name, h, D)


if __name__ == "__main__":
    try:
        log.info("-" * 31 + "  start  " + "-" * 31)
        experiments()
        log.info("-" * 32 + "  end  " + "-" * 32)
    except Exception as e:
        log.error(e)
        print_exception(e)
