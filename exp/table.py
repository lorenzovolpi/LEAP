import itertools as IT
import os

import pandas as pd

from exp.config import (
    PROBLEM,
    get_acc_names,
    get_baseline_names,
    get_classifier_names,
    get_dataset_names,
    get_method_names,
    root_dir,
)
from exp.util import load_results, rename_datasets, rename_methods
from leap.table import Format, Table

method_map = {
    "Naive": 'Na\\"ive',
    "ATC-MC": "ATC",
    "LEAP(ACC-MLP)": "LEAP$_{\\mathrm{ACC}}$",
    "LEAP(KDEy-MLP)": "LEAP$_{\\mathrm{KDEy}}$",
    "PHD(KDEy-MLP)": "LEAP(PPS)$_{\\mathrm{KDEy}}$",
    "OCE(KDEy-MLP)-SLSQP": "OLEAP$_{\\mathrm{KDEy}}$",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def tables():
    res = load_results()

    def gen_table(df: pd.DataFrame, name, datasets, methods, baselines):
        tbl = Table(name=name, benchmarks=datasets, methods=methods, baselines=baselines)
        tbl.format = Format(
            mean_prec=3,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            color=True,
            color_mode="baselines",
            simple_stat=True,
            best_color="OliveGreen",
            mid_color="SeaGreen",
        )
        tbl.format.mean_macro = True
        for dataset, method in IT.product(datasets, methods):
            values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), "acc_err"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    classifiers = get_classifier_names()
    datasets = get_dataset_names()
    methods = get_method_names(with_oracle=False)
    baselines = get_baseline_names()
    accs = get_acc_names()

    tbls = []
    for classifier, acc in IT.product(classifiers, accs):
        _df = res.loc[(res["classifier"] == classifier) & (res["acc_name"] == acc), :]
        name = f"{PROBLEM}_{classifier}_{acc}"
        _df, _datasets = rename_datasets(dataset_map, _df, datasets)
        _df, _methods, _baselines = rename_methods(method_map, _df, methods, baselines)
        tbls.append(gen_table(_df, name, _datasets, _methods, _baselines))

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")
    Table.LatexPDF(pdf_path, tables=tbls, landscape=False)


def leap_true_solve():
    res = load_results()
    methods = ["LEAP(KDEy)", "LEAP(KDEy-a)", "LEAP(MDy)"]
    md_path = os.path.join(root_dir, "tables", f"{PROBLEM}_true_solve.md")

    pd.pivot_table(
        res.loc[res["method"].isin(methods)],
        columns=["classifier", "method"],
        index=["dataset"],
        values="true_solve",
    ).to_markdown(md_path)


if __name__ == "__main__":
    tables()
    # leap_true_solve()
