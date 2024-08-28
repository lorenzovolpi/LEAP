import itertools as IT
import os
from traceback import print_exception

import quapy as qp
from quapy.protocol import UPP

from leap.experiments.generators import (
    gen_acc_measure,
    gen_bin_datasets,
    gen_classifiers,
    gen_methods,
    gen_multi_datasets,
    gen_product,
    get_method_names,
)
from leap.experiments.report import TestReport
from leap.experiments.util import (
    fit_or_switch,
    get_logger,
    get_plain_prev,
    get_predictions,
    prevs_from_prot,
)
from leap.utils.commons import save_dataset_stats, true_acc

PROBLEM = "binary"
ORACLE = False
basedir = PROBLEM + ("-oracle" if ORACLE else "")

NUM_TEST = 1000

if PROBLEM == "binary":
    qp.environ["SAMPLE_SIZE"] = 100
    gen_datasets = gen_bin_datasets
elif PROBLEM == "multiclass":
    qp.environ["SAMPLE_SIZE"] = 100
    gen_datasets = gen_multi_datasets


log = get_logger()


def all_exist_pre_check(basedir, cls_name, dataset_name):
    method_names = get_method_names()
    acc_names = [acc_name for acc_name, _ in gen_acc_measure()]

    all_exist = True
    for method, acc in IT.product(method_names, acc_names):
        path = TestReport(basedir, cls_name, acc, dataset_name, None, None, method).get_path()
        all_exist = os.path.exists(path)
        if not all_exist:
            break

    return all_exist


def experiments():
    log.info("-" * 31 + "  start  " + "-" * 31)

    for (cls_name, h), (dataset_name, (L, V, U)) in gen_product(gen_classifiers, gen_datasets):
        if all_exist_pre_check(basedir, cls_name, dataset_name):
            log.info(f"{cls_name} on dataset={dataset_name}: all results already exist, skipping")
            continue

        log.info(f"{cls_name} training on dataset={dataset_name}")
        h.fit(*L.Xy)

        # test generation protocol
        test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)

        # compute some stats of the dataset
        save_dataset_stats(dataset_name, test_prot, L, V)

        # precompute the actual accuracy values

        true_accs = {}
        for acc_name, acc_fn in gen_acc_measure(multiclass=True):
            true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

        L_prev = get_plain_prev(L.prevalence())
        for method_name, method, with_oracle in gen_methods(h):
            V_prev = get_plain_prev(V.prevalence())

            t_train = None
            for acc_name, acc_fn in gen_acc_measure(multiclass=True):
                report = TestReport(
                    basedir,
                    cls_name,
                    acc_name,
                    dataset_name,
                    L_prev,
                    V_prev,
                    method_name,
                )
                if os.path.exists(report.get_path()):
                    log.info(f"{method_name}: {acc_name} exists, skipping")
                    continue

                try:
                    method, _t_train = fit_or_switch(method, V, acc_fn, t_train is not None)
                    t_train = t_train if _t_train is None else _t_train

                    test_prevs = prevs_from_prot(test_prot)
                    estim_accs, t_test_ave = get_predictions(method, test_prot, with_oracle)
                    report.add_result(test_prevs, true_accs[acc_name], estim_accs, t_train, t_test_ave)
                except Exception as e:
                    print_exception(e)
                    log.warning(f"{method_name}: {acc_name} gave error '{e}' - skipping")
                    continue

                report.save_json(basedir, acc_name)

                log.info(f"{method_name}: {acc_name} done [t_train:{t_train:.3f}s; t_test_ave:{t_test_ave:.3f}s]")

    log.info("-" * 32 + "  end  " + "-" * 32)


if __name__ == "__main__":
    try:
        experiments()
    except Exception as e:
        log.error(e)
        print_exception(e)
