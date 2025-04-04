import itertools as IT
import os

import pandas as pd
from quacc.table import Format, Table

from exp.leap.config import (
    PROBLEM,
    get_acc_names,
    get_baseline_names,
    get_classifier_names,
    get_dataset_names,
    root_dir,
)
from exp.leap.util import load_results, rename_datasets, rename_methods

method_map = {
    "Naive": 'Na\\"ive',
    "ATC-MC": "ATC",
    "LEAP(ACC)": "\\leapacc",
    "LEAP(KDEy)": "\\leapplus",
    "S-LEAP(KDEy)": "\\leapppskde",
    "O-LEAP(KDEy)-SLSQP": "\\oleapkde",
}

dataset_map = {
    "poker_hand": "poker-hand",
    "hand_digits": "hand-digits",
    "page_block": "page-block",
    "image_seg": "image-seg",
}


def decorate_datasets(df, datasets):
    def _decorate(d):
        return r"\textsf{" + d + r"}"

    _datasets = [_decorate(d) for d in datasets]
    for d in df["dataset"].unique():
        df.loc[df["dataset"] == d, "dataset"] = _decorate(d)
    return df, _datasets


def tables():
    classifiers = get_classifier_names()
    datasets = get_dataset_names()
    baselines = get_baseline_names()
    methods = baselines + ["LEAP(ACC)", "LEAP(KDEy)", "S-LEAP(KDEy)", "O-LEAP(KDEy)"]
    accs = get_acc_names()

    res = load_results(filter_methods=methods)

    def gen_table(df: pd.DataFrame, name, datasets, methods, baselines):
        tbl = Table(name=name, benchmarks=datasets, methods=methods, baselines=baselines)
        tbl.format = Format(
            mean_prec=3,
            show_std=True,
            remove_zero=True,
            with_rank_mean=False,
            with_mean=True,
            mean_macro=False,
            color=True,
            color_mode="local",
            simple_stat=True,
            best_color="green",
        )
        for dataset, method in IT.product(datasets, methods):
            values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), "acc_err"].to_numpy()
            for v in values:
                tbl.add(dataset, method, v)
        return tbl

    tbls = []
    for classifier, acc in IT.product(classifiers, accs):
        _df = res.loc[(res["classifier"] == classifier) & (res["acc_name"] == acc), :]
        name = f"{PROBLEM}_{classifier}_{acc}"
        _df, _datasets = rename_datasets(dataset_map, _df, datasets)
        _df, _datasets = decorate_datasets(_df, _datasets)
        _df, _methods, _baselines = rename_methods(method_map, _df, methods, baselines)
        tbls.append(gen_table(_df, name, _datasets, _methods, _baselines))

    pdf_path = os.path.join(root_dir, "tables", f"{PROBLEM}.pdf")
    new_commands = [
        "\\newcommand{\leapacc}{LEAP$_\\mathrm{ACC}$}",
        "\\newcommand{\leapplus}{LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\leapppskde}{S-LEAP$_\\mathrm{KDEy}$}",
        "\\newcommand{\oleapkde}{O-LEAP$_\\mathrm{KDEy}$}",
    ]
    Table.LatexPDF(pdf_path, tables=tbls, landscape=False, new_commands=new_commands)


if __name__ == "__main__":
    tables()
