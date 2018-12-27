import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

FLOAT_TYPE = np.float32


def get_freq(x):
    if len(x) == 0:
        return np.NaN
    return pd.Series(x).value_counts().idxmax()


def map_arr(x, value_map):
    return pd.Series(x).map(value_map).values


def value_counts(x):
    return pd.Series(x).value_counts().sort_index()


def calc_post_proba(x, y):
    return pd.DataFrame({"x": x, "y": y}).groupby("x").y.mean()


def calc_woe(x, y, alpha=0):
    pos_total = y.sum()
    neg_total = len(y) - pos_total
    pos_rate = float(pos_total) / len(y)
    neg_rate = 1 - pos_rate
    pos_add = pos_rate * alpha if alpha > 0 else 0
    neg_add = neg_rate * alpha if alpha > 0 else 0
    regroup = pd.DataFrame({"x": x, "pos": y, "neg": 1 - y}).groupby("x").sum()
    regroup["pos_rate"] = (regroup.pos + pos_add) / (pos_total + pos_add * regroup.shape[0])
    regroup["neg_rate"] = (regroup.neg + neg_add) / (neg_total + neg_add * regroup.shape[0])
    regroup["woe"] = regroup.apply(lambda x: np.log(x.pos_rate * 1.0 / x.neg_rate), axis=1).astype(FLOAT_TYPE)
    return regroup


class WOEEncoder:

    def __init__(self, alpha=0, default_value=0.0, use_default_class=True):
        self.alpha_ = alpha
        self.default_value_ = default_value
        self.use_default_class_ = use_default_class
        self.woe_maps_ = {}
        self.iv_map_ = {}

    def fit(self, X, y):
        for col_ in range(X.shape[1]):
            col_arr = X[:, col_]
            woe_data = calc_woe(col_arr, y, self.alpha_)
            iv = ((woe_data.pos_rate - woe_data.neg_rate) * woe_data.woe).sum()
            self.woe_maps_[col_] = woe_data
            self.iv_map_[col_] = iv
        return self

    def transform(self, X):
        X_copy = np.zeros(X.shape).astype(np.float32)
        # print(X_copy.shape)
        for col_, map_ in self.woe_maps_.items():
            col_arr = X[:, col_]
            X_copy[:, col_] = map_arr(col_arr, map_.woe)
        X_copy[np.isnan(X_copy)] = self.default_value_
        return X_copy

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


def calc_woe_mp(argv):
    x = argv["x"]
    y = argv["y"]
    col_ = argv["col_"]
    alpha_ = argv["alpha_"]
    woe_data = calc_woe(x, y, alpha_)
    iv = ((woe_data.pos_rate - woe_data.neg_rate) * woe_data.woe).sum()
    return (col_, woe_data, iv)


class MPWOEEncoder(WOEEncoder):
    ''' multiprocessing version of WOEEncoder '''

    def __init__(self, alpha=0, n_jobs=-1):
        WOEEncoder.__init__(self, alpha)
        self.n_jobs_ = -1

    def fit(self, X, y):
        param_list = [
            {
                # "X": X,
                "x": X[:, col_],
                "y": y,
                "col_": col_,
                "alpha_": self.alpha_
            } for col_ in range(X.shape[1])
        ]

        cpu_cnt = cpu_count()
        processes_cnt = cpu_cnt if self.n_jobs_ < 0 or self.n_jobs_ > cpu_cnt else self.n_jobs_
        pool = Pool(processes=processes_cnt)
        result = pool.map(calc_woe_mp, param_list)
        pool.close()
        pool.join()

        self.woe_maps_ = dict([(x[0], x[1]) for x in result])
        self.iv_map_ = dict([(x[0], x[2]) for x in result])
        return self


def replace_null(arr, unknown_token_num=-1, unknown_token_str="NaN", known_set=None):
    series = pd.Series(arr)
    token = unknown_token_str if series.dtype == "O" else unknown_token_num
    if known_set is not None:
        series[~series.isin(known_set)] = token
    return series.replace(np.NaN, token)


def get_value_map(arr, default=-1):
    values = pd.Series(arr).unique()
    value_map = dict(zip(values, range(values.shape[0])))
    value_map[np.NaN] = default
    value_map["nan"] = default
    value_map["XNA"] = default
    return value_map


class MyLabelEncoder():
    ''' MyLabelEncoder, which can handle null values and stranger class'''

    def __init__(self, default=-1):
        self.value_map_ = None
        self.default = -1

    def fit_transform(self, x):
        self.value_map_ = get_value_map(x, default=self.default)
        return self.transform(x)

    def transform(self, x):
        result = map_arr(x, self.value_map_)
        result[np.isnan(result)] = self.default
        return result


def kfold_normalize(X, y, group, encoder):
    woe_oof = np.zeros(X.shape)
    for grp in sorted(np.unique(group)):
        encoder.fit(X[group != grp], y[group != grp])
        woe_oof[group == grp] = encoder.transform(X[group == grp])
    return woe_oof
