import json
import os
import urllib.request
from collections import defaultdict

import numpy as np
import quapy as qp
from quapy.data import LabelledCollection

import leap

RCV1_HIERARCHY_URL = "http://www.ai.mit.edu/projects/jmlr/papers/volume5/lewis04a/a03-expanded-topics-hierarchy/rcv1.topics.hier.expanded"


def split_train(train: LabelledCollection, train_val_split: float):
    return train.split_stratified(train_prop=train_val_split, random_state=qp.environ["_R_SEED"])


def get_rcv1_class_info():
    os.makedirs(leap.env["LEAP_DATA"], exist_ok=True)
    json_path = os.path.join(leap.env["LEAP_DATA"], "rcv1_class_info.json")
    if not os.path.exists(json_path):
        # retrieve hierarchy file and class names
        hierarchy_tmp_path = os.path.join(leap.env["LEAP_DATA"], "rcv1_hierarchy.tmp")
        urllib.request.urlretrieve(RCV1_HIERARCHY_URL, filename=hierarchy_tmp_path)
        tree = defaultdict(lambda: [])
        class_names = set()
        with open(hierarchy_tmp_path, "r") as tf:
            lines = tf.readlines()
        for line in lines:
            tokens = [s for s in line.strip().split(" ") if len(s) > 0]
            parent, child = tokens[1], tokens[3]
            if parent != "None":
                tree[parent].append(child)
                class_names.add(child)
            class_names.add("Root")

        # sort class names
        class_names = sorted(list(class_names))

        with open(json_path, "w") as jf:
            json.dump(
                {
                    "class_names": class_names,
                    "tree": tree,
                },
                jf,
                indent=2,
            )

        if os.path.exists(hierarchy_tmp_path):
            os.remove(hierarchy_tmp_path)

    with open(json_path, "r") as jf:
        class_info = json.load(jf)

    class_names = np.array(class_info["class_names"])
    tree, index = {}, {}
    for parent, children in class_info["tree"].items():
        children = np.array(children)
        idxs = np.where(np.in1d(class_names, children))[0]
        if len(idxs) == len(children):
            tree[parent] = children
            index[parent] = idxs

    return class_names.tolist(), tree, index
