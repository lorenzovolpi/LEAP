import os

env = {
    "OUT_DIR": os.getenv("PHD_OUT_DIR", "."),
    "QUAPY_DATA": os.getenv("PHD_QUAPY_DATA", os.path.expanduser("~/quapy_data")),
}
