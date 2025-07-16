import quapy as qp

import leap

PROJECT = "leap"
root_dir = leap.env["OUT_DIR"]
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
CSV_SEP = ","

_valid_problems = ["binary", "multiclass"]
PROBLEM = "multiclass"
