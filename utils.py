import warnings
# warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import time
import yaml
import json
import addict
import random
import numpy as np
from contextlib import contextmanager
from sklearn.externals import joblib


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def dict_rename(src_dict, src_key, tar_key):
    tar_dict = src_dict.copy()
    if src_key in tar_dict:
        tar_dict[tar_key] = tar_dict[src_key]
        del tar_dict[src_key]
    return tar_dict


def dict_filter(src_dict, keys):
    src_dict = addict.Dict(src_dict)
    tar_dict = addict.Dict()
    for key in keys:
        tar_dict[key] = src_dict[key]
    return tar_dict


def save_obj(data, filepath):
    joblib.dump(data, filepath)


def load_obj(filepath):
    return joblib.load(filepath)


def load_json(filepath):
    with(open(filepath)) as f:
        data = json.load(f)
    return data


def save_json(data, filepath):
    with(open(filepath, "w")) as f:
        json.dump(data, f, indent=4)


def load_yaml(filepath):
    with(open(filepath)) as f:
        config = yaml.load(f)
    return addict.Dict(config)


def save_yaml(data, filepath):
    with(open(filepath, "w")) as f:
        yaml.dump(data, f)


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
