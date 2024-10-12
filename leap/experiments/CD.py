import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from Orange.evaluation import scoring

import leap
from leap.experiments.generators import gen_acc_measure, gen_bin_datasets, gen_multi_datasets
from leap.experiments.report import Report

PROBLEM = "multiclass"
ERROR = leap.error.ae

METHODS = ["Naive", "ATC", "DoC", "LEAP", "LEAP-plus"]
CLASSIFIERS = ["LR", "KNN", "SVM", "MLP"]

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets
elif PROBLEM == "binary_ext":
    gen_datasets = gen_bin_datasets
    METHODS += ["LEAPcc", "NaiveRescaling", "NaiveRescaling-plus"]
elif PROBLEM == "binary_oracle":
    gen_datasets = gen_bin_datasets
    METHODS += ["LEAPcc", "NaiveRescaling", "NaiveRescaling-plus", "LEAP-oracle", "NaiveRescaling-oracle"]
elif PROBLEM == "multiclass":
    gen_datasets = gen_multi_datasets

BENCHMARKS = [name for name, _ in gen_datasets(only_names=True)]
ACC_NAMES = [acc_name for acc_name, _ in gen_acc_measure()]


def get_results_problem(problem):
    problems = {
        "binary_oracle": "binary",
        "binary_ext": "binary",
    }
    return problems.get(problem, problem)


def scikit():
    for acc_name in ACC_NAMES:
        for cls_name in CLASSIFIERS:
            plt_dir = os.path.join(os.path.expanduser("~/leap/plots/CD"), acc_name)
            os.makedirs(plt_dir, exist_ok=True)

            plt_path = os.path.join(plt_dir, cls_name + ".png")
            rep = Report.load_results(get_results_problem(PROBLEM), cls_name, acc_name, BENCHMARKS, METHODS)
            data = rep.table_data(mean=False, error=ERROR)

            print(data)

            test_results = sp.posthoc_wilcoxon(
                data,
                group_col="method",
                val_col="acc_err",
            )
            print(test_results)

            avg_rank = data.groupby("dataset").acc_err.rank(pct=True).groupby(data.method).mean()
            print(avg_rank)

            plt.figure(figsize=(10, 2), dpi=100)
            plt.title(cls_name)
            sp.critical_difference_diagram(avg_rank, test_results)
            # plt.show()
            plt.savefig(plt_path, bbox_inches="tight", dpi=300)

    # naive
    # atc
    # doc
    # leapcc
    # leap
    # leap+
    # leap_oracle
    #
    # rng = np.random.default_rng(1)
    # dict_data = {
    #     "model1": rng.normal(loc=0.2, scale=0.1, size=30),
    #     "model2": rng.normal(loc=0.2, scale=0.1, size=30),
    #     "model3": rng.normal(loc=0.4, scale=0.1, size=30),
    #     "model4": rng.normal(loc=0.5, scale=0.1, size=30),
    #     "model5": rng.normal(loc=0.7, scale=0.1, size=30),
    #     "model6": rng.normal(loc=0.7, scale=0.1, size=30),
    #     "model7": rng.normal(loc=0.8, scale=0.1, size=30),
    #     "model8": rng.normal(loc=0.9, scale=0.1, size=30),
    # }
    # data = (
    #     pd.DataFrame(dict_data)
    #     .rename_axis("cv_fold")
    #     .melt(var_name="estimator", value_name="score", ignore_index=False)
    #     .reset_index()
    # )
    # print(data)
    # test_results = sp.posthoc_wilcoxon(
    #     data,
    #     group_col="estimator",
    #     val_col="score",
    # )
    # print(test_results)
    # avg_rank = data.groupby("cv_fold").score.rank(pct=True).groupby(data.estimator).mean()
    # print(avg_rank)
    # plt.figure(figsize=(10, 2), dpi=100)
    # plt.title("CD")
    # sp.critical_difference_diagram(avg_rank, test_results)
    # plt.show()


def orange():
    avranks = [1.9, 3.2, 2.8, 3.3]
    cd = scoring.compute_CD(avranks, 30)  # tested on 30 datasets
    print(cd)


if __name__ == "__main__":
    orange()
