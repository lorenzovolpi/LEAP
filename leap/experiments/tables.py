import itertools as IT
import os
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd

import leap
from leap.experiments.generators import (
    gen_acc_measure,
    gen_bin_datasets,
    gen_multi_datasets,
)
from leap.experiments.report import Report
from leap.table import Format, Table

PROBLEM = "binary"
ERROR = leap.error.ae

METHODS = ["Naive", "ATC", "DoC", "LEAP", "LEAP-plus"]
CLASSIFIERS = ["LR", "KNN", "SVM", "MLP"]

if PROBLEM == "binary":
    gen_datasets = gen_bin_datasets
elif PROBLEM == "binary_ext":
    gen_datasets = gen_bin_datasets
    METHODS = ["Naive", "ATC", "DoC", "LEAPcc", "LEAP", "LEAP-plus", "LEAP-oracle"]
    CLASSIFIERS = ["LR", "MLP"]
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
        "binary_paper": "binary",
    }
    return problems.get(problem, problem)


def rename_method(m):
    methods_dict = {
        "Naive": "\\naive",
        "LEAP": "\\phdacc",
        "LEAP-plus": "\\phdplus",
        "LEAP-oracle": "\\phdoracle",
        "LEAPcc": "\\phdcc",
    }
    return methods_dict.get(m, m)


def rename_cls(cls):
    cls_dict = {"KNN": "$k$NN"}
    return cls_dict.get(cls, cls)


def table_from_df(df: pd.DataFrame, name, benchmarks, methods) -> Table:
    tbl = Table(name=name, benchmarks=benchmarks, methods=methods)
    tbl.format = Format(mean_prec=3, show_std=False, remove_zero=True, with_rank_mean=False, with_mean=True)
    tbl.format.mean_macro = False
    for dataset, method in IT.product(benchmarks, methods):
        values = df.loc[(df["dataset"] == dataset) & (df["method"] == method), ["acc_err"]].to_numpy()
        for v in values:
            tbl.add(dataset, method, v)

    return tbl


def get_method_cls_name(method, cls):
    return f"{method}-{cls}"


def hstack_tables(tables, pdf_path):
    tbl_dict = defaultdict(lambda: [])
    tbl_endline = defaultdict(lambda: "\\\\")
    for t in tables:
        tabular = StringIO(t.tabular())
        for ln in tabular:
            ln = ln.strip()

            if ln.startswith("\\begin{tabular}"):
                continue
            if ln.startswith("\\end{tabular}"):
                continue
            if ln.startswith("\\cline"):
                continue
            if ln.startswith("\\multicolumn"):
                continue

            amp_idx = ln.find("&")
            name = ln[:amp_idx].strip()
            row = ln[amp_idx + 1 :]
            if row.endswith("\\\\"):
                row = row[:-2].strip()
            elif row.endswith("\\\\\\hline"):
                row = row[:-8].strip()
                tbl_endline[name] = "\\\\\\hline"
            tbl_dict[name].append(row)

    corpus = []
    for name, row_l in tbl_dict.items():
        row = " & ".join(row_l)
        corpus.append(f"{name} & {row} {tbl_endline[name]}")
    corpus = "\n".join(corpus) + "\n"
    header = "\\setlength{\\tabcolsep}{0pt}\n"
    begin = "\\begin{tabular}{|c|" + (("c" * len(METHODS)) + "|") * len(CLASSIFIERS) + "}\n"
    end = "\\end{tabular}\n"
    cline = "\\cline{2-" + str(len(METHODS) * len(CLASSIFIERS) + 1) + "}\n"
    multicol1 = (
        "\\multicolumn{1}{c|}{} & "
        + " & ".join(["\\multicolumn{" + str(len(METHODS)) + "}{c|}{" + rename_cls(cls) + "}" for cls in CLASSIFIERS])
        + " \\\\\n"
    )
    tbl_methods = [rename_method(m) for m in METHODS]
    multicol2 = (
        "\\multicolumn{1}{c|}{} & "
        + " & ".join([" & ".join(["\\side{" + m + "}" for m in tbl_methods]) for _ in CLASSIFIERS])
        + " \\\\\\hline\n"
    )

    hstack_dir = os.path.join(str(Path(pdf_path).parent), "tables")
    os.makedirs(hstack_dir, exist_ok=True)
    hstack_path = os.path.join(hstack_dir, f"{PROBLEM}_hstack.tex")
    with open(hstack_path, "w") as f:
        f.write(header + begin + cline + multicol1 + cline + multicol2 + corpus + end)


def gen_n2e_tables():
    pdf_path = f"tables/{PROBLEM}.pdf"

    tables = []
    for acc_name in ACC_NAMES:
        for cls_name in CLASSIFIERS:
            rep = Report.load_results(get_results_problem(PROBLEM), cls_name, acc_name, BENCHMARKS, METHODS)
            df = rep.table_data(mean=False, error=ERROR)

            cls_name = rename_cls(cls_name)

            # build table
            tbl_name = f"{PROBLEM}_{cls_name}_{acc_name}"
            tbl = table_from_df(df, name=tbl_name, benchmarks=BENCHMARKS, methods=METHODS)
            tables.append(tbl)

    Table.LatexPDF(pdf_path=pdf_path, tables=tables, landscape=False)
    hstack_tables(tables, pdf_path)


if __name__ == "__main__":
    gen_n2e_tables()
