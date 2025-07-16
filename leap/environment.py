import os

env = {
    "OUT_DIR": os.getenv("LEAP_OUT_DIR", "./output"),
    "QUAPY_DATA": os.getenv("LEAP_QUAPY_DATA", os.path.expanduser("~/quapy_data")),
    "SKLEARN_DATA": os.getenv("LEAP_SKLEARN_DATA", os.path.expanduser("~/scikit_learn_data")),
    "N_JOBS": int(os.getenv("LEAP_N_JOBS", -2)),
    "FORCE_NJOBS": int(os.getenv("LEAP_FORCE_NJOBS", 1)) > 0,
}
