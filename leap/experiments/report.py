import itertools
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

import leap
from leap.utils.commons import get_results_path, load_json_file, save_json_file


class TestReport:
    def __init__(
        self,
        basedir,
        cls_name,
        acc_name,
        dataset_name,
        train_prev,
        val_prev,
        method_name,
    ):
        self.basedir = basedir
        self.cls_name = cls_name
        self.acc_name = acc_name
        self.dataset_name = dataset_name
        self.train_prev = train_prev
        self.val_prev = val_prev
        self.method_name = method_name

    def get_path(self):
        return get_results_path(
            self.basedir,
            self.cls_name,
            self.acc_name,
            self.dataset_name,
            self.method_name,
        )

    def add_result(self, test_prevs, true_accs, estim_accs, t_train, t_test_ave):
        self.test_prevs = test_prevs
        self.estim_accs = estim_accs
        self.true_accs = true_accs
        self.t_train = t_train
        self.t_test_ave = t_test_ave

        return self

    def save_json(self, basedir, acc_name):
        result = {
            "basedir": self.basedir,
            "cls_name": self.cls_name,
            "acc_name": self.acc_name,
            "dataset_name": self.dataset_name,
            "train_prev": self.train_prev,
            "val_prev": self.val_prev,
            "method_name": self.method_name,
            "test_prevs": self.test_prevs,
            "true_accs": self.true_accs,
            "estim_accs": self.estim_accs,
            "t_train": self.t_train,
            "t_test_ave": self.t_test_ave,
        }

        save_json_file(self.get_path(), result)

    @classmethod
    def load_json(cls, path) -> "TestReport":
        def _test_report_hook(_dict):
            return TestReport(
                basedir=_dict["basedir"],
                cls_name=_dict["cls_name"],
                acc_name=_dict["acc_name"],
                dataset_name=_dict["dataset_name"],
                train_prev=_dict["train_prev"],
                val_prev=_dict["val_prev"],
                method_name=_dict["method_name"],
            ).add_result(
                test_prevs=_dict["test_prevs"],
                true_accs=_dict["true_accs"],
                estim_accs=_dict["estim_accs"],
                t_train=_dict["t_train"],
                t_test_ave=_dict["t_test_ave"],
            )

        return load_json_file(path, object_hook=_test_report_hook)


class Report:
    def __init__(self, results: dict[str, list[TestReport]]):
        self.results = results

    @classmethod
    def load_results(cls, basedir, cls_name, acc_name, dataset_name="*", method_name="*") -> "Report":
        _results = defaultdict(lambda: [])
        if isinstance(method_name, str):
            method_name = [method_name]
        if isinstance(dataset_name, str):
            dataset_name = [dataset_name]
        for dataset_, method_ in itertools.product(dataset_name, method_name):
            path = get_results_path(basedir, cls_name, acc_name, dataset_, method_)
            for file in glob(path):
                if file.endswith(".json"):
                    method = Path(file).stem
                    _res = TestReport.load_json(file)
                    _results[method].append(_res)
        return Report(_results)

    def table_data(self, mean=True, error=leap.error.ae):
        assert error in leap.error.ACCURACY_ERROR_SINGLE, "Unknown error function"
        dfs = []
        for _method, _results in self.results.items():
            _dataset_map = defaultdict(lambda: [])
            for r in _results:
                _dataset_map[r.dataset_name].append(r)

            for _dataset, _res in _dataset_map.items():
                _true_acc = np.hstack([_r.true_accs for _r in _res])
                _estim_acc = np.hstack([_r.estim_accs for _r in _res])
                _acc_err = error(_true_acc, _estim_acc)
                report_df = pd.DataFrame(
                    np.vstack([_true_acc, _estim_acc, _acc_err]).T,
                    columns=["true_accs", "estim_accs", "acc_err"],
                )
                report_df.loc[:, "method"] = np.tile(_method, (len(report_df),))
                report_df.loc[:, "dataset"] = np.tile(_dataset, (len(report_df),))
                dfs.append(report_df)

        all_df = pd.concat(dfs, axis=0, ignore_index=True)
        if mean:
            all_df = all_df.groupby(["method", "dataset"]).mean().reset_index()
        return all_df

    def diagonal_plot_data(self):
        dfs = []
        for _method, _results in self.results.items():
            _true_acc = np.hstack([_r.true_accs for _r in _results])
            _estim_acc = np.hstack([_r.estim_accs for _r in _results])
            _prev = np.hstack([_r.test_prevs for _r in _results])
            method_df = pd.DataFrame(
                np.vstack([_true_acc, _estim_acc, _prev]).T,
                columns=["true_accs", "estim_accs", "prev"],
            )
            method_df.loc[:, "method"] = np.tile(_method, (len(method_df),))
            dfs.append(method_df)

        return pd.concat(dfs, axis=0, ignore_index=True)
