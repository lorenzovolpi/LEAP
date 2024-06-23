import os

env = {
    "OUT_DIR": os.getenv("LEAP_OUT_DIR", "."),
    "QUAPY_DATA": os.getenv("LEAP_QUAPY_DATA", os.path.expanduser("~/quapy_data")),
}
