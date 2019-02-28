import glob
import gc
import pandas as pd
import numpy as np
from utils import timer
from filepath_collection import gen_submission_filepath, gen_oof_filepath, features_downcast
from downcast import load_dataframe32, save_dataframe32


def get_train_downcast():
    return "./data/features_downcast/train"


def get_test_downcast():
    return "./data/features_downcast/test"


def get_hist_trans_downcast():
    lines = glob.glob(
        "./data/features_downcast/historical_transactions*.data64.npy")
    return [line.replace(".data64.npy", "").replace("\\", "/") for line in lines]


def get_new_trans_downcast():
    lines = glob.glob(
        "./data/features_downcast/new_merchant_transactions*.data64.npy")
    return [line.replace(".data64.npy", "").replace("\\", "/") for line in lines]


def get_all_trans_downcast():
    lines = glob.glob(
        "./data/features_downcast/all_transactions_transactions*.data64.npy")
    return [line.replace(".data64.npy", "").replace("\\", "/") for line in lines]


def get_newk():
    return features_downcast("newk")


def get_psum():
    return features_downcast("monthly_psum")


def rename_columns(data, header):
    def append_header(col):
        if col == "card_id":
            return col
        return "{}_{}".format(col, header)

    data.columns = [append_header(col) for col in data.columns]
    return data


def calc_yearmonth(series):
    return series.dt.year * 100 + series.dt.month


def append_time_features(data):
    data["first_active_month"] = pd.to_datetime(data.first_active_month)
    data["fa_month"] = calc_yearmonth(data.first_active_month).astype(np.float32)
    del data["first_active_month"]
    gc.collect()
    return data


def load_train():
    return load_dataframe32(get_train_downcast())


def load_test():
    return load_dataframe32(get_test_downcast())


def load_train_features():
    data = load_dataframe32(get_train_downcast())
    data = append_time_features(data)
    for filepath in get_all_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "all_trans")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(data.shape)
    for filepath in get_hist_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "hist_trans")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(data.shape)
    for filepath in get_new_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "new_trans")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(data.shape)
    return data


def load_test_features():
    data = load_dataframe32(get_test_downcast())
    data = append_time_features(data)
    for filepath in get_all_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "all_trans")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(data.shape)
    for filepath in get_hist_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "hist_trans")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(data.shape)
    for filepath in get_new_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "new_trans")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(data.shape)
    return data


def load_train_newk_features(nrows=None):
    with timer("train"):
        data = load_dataframe32(get_train_downcast())
        if nrows is not None and nrows > 0:
            data = data[:nrows].copy()
            gc.collect()
        data = append_time_features(data)
    with timer("all_trans"):
        for filepath in get_all_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "all_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("hist_trans"):
        for filepath in get_hist_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "hist_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("new_trans"):
        for filepath in get_new_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "new_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("newk"):
        part = load_dataframe32(get_newk())
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
        # print(data.shape)

    return data


def load_test_newk_features(nrows=None):
    with timer("test"):
        data = load_dataframe32(get_test_downcast())
        if nrows is not None and nrows > 0:
            data = data[:nrows].copy()
            gc.collect()
        data = append_time_features(data)
    with timer("all_trans"):
        for filepath in get_all_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "all_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("hist_trans"):
        for filepath in get_hist_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "hist_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("new_trans"):
        for filepath in get_new_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "new_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("newk"):
        part = load_dataframe32(get_newk())
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
        # print(data.shape)

    return data


def load_train_psum_features(nrows=None):
    with timer("train"):
        data = load_dataframe32(get_train_downcast())
        if nrows is not None and nrows > 0:
            data = data[:nrows].copy()
            gc.collect()
        data = append_time_features(data)
    with timer("all_trans"):
        for filepath in get_all_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "all_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("hist_trans"):
        for filepath in get_hist_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "hist_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("new_trans"):
        for filepath in get_new_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "new_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("newk"):
        part = load_dataframe32(get_newk())
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
        # print(data.shape)
    with timer("psum"):
        part = load_dataframe32(get_psum())
        part = rename_columns(part, "psum")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    return data


def load_test_psum_features(nrows=None):
    with timer("test"):
        data = load_dataframe32(get_test_downcast())
        if nrows is not None and nrows > 0:
            data = data[:nrows].copy()
            gc.collect()
        data = append_time_features(data)
    with timer("all_trans"):
        for filepath in get_all_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "all_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("hist_trans"):
        for filepath in get_hist_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "hist_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("new_trans"):
        for filepath in get_new_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "new_trans")
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
        # print(data.shape)
    with timer("newk"):
        part = load_dataframe32(get_newk())
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
        # print(data.shape)
    with timer("psum"):
        part = load_dataframe32(get_psum())
        part = rename_columns(part, "psum")
        data = data.merge(part, how="left",
                          left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    return data


def load_train_all_features(nrows=None, mode=0):
    data = load_train_psum_features(nrows)

    for name in ("monthly_pmax",
                 "monthly_merchant_pmax",
                 "main_merchant_count",
                 "monthly_merchant_avg_std"):

        with timer(name):
            part = load_dataframe32(features_downcast(name))
            part = rename_columns(part, name)
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
    if mode > 0:
        for name in ("monthly_pmax_abs",
                     "monthly_merchant_pmax_abs",
                     "duar_count",
                     "monthly_merchant_avg_std_abs"):
            with timer(name):
                part = load_dataframe32(features_downcast(name))
                part = rename_columns(part, name)
                data = data.merge(part, how="left",
                                  left_on="card_id", right_on="card_id")
    return data


def load_test_all_features(nrows=None, mode=0):
    data = load_test_psum_features(nrows)

    for name in ("monthly_pmax",
                 "monthly_merchant_pmax",
                 "main_merchant_count",
                 "monthly_merchant_avg_std"):

        with timer(name):
            part = load_dataframe32(features_downcast(name))
            part = rename_columns(part, name)
            data = data.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
    if mode > 0:
        for name in ("monthly_pmax_abs",
                     "monthly_merchant_pmax_abs",
                     "duar_count",
                     "monthly_merchant_avg_std_abs"):
            with timer(name):
                part = load_dataframe32(features_downcast(name))
                part = rename_columns(part, name)
                data = data.merge(part, how="left",
                                  left_on="card_id", right_on="card_id")
    return data


def load_train_test_all(nrows=None,
                        with_test=True,
                        names=("newk",
                               "monthly_psum",
                               "monthly_pmax",
                               "monthly_merchant_pmax",
                               "main_merchant_count",
                               "monthly_merchant_avg_std",
                               "monthly_pmax_abs",
                               "monthly_merchant_pmax_abs",
                               "duar_count",
                               "monthly_merchant_avg_std_abs")):
    with timer("train"):
        train = load_dataframe32(get_train_downcast())
        if nrows is not None and nrows > 0:
            train = train[:nrows].copy()
            gc.collect()
        train = append_time_features(train)

    if with_test:
        with timer("test"):
            test = load_dataframe32(get_test_downcast())
            if nrows is not None and nrows > 0:
                test = test[:nrows].copy()
                gc.collect()
            test = append_time_features(test)
    else:
        test = pd.DataFrame()
        test["card_id"] = []

    with timer("hist_trans"):
        for filepath in get_hist_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "hist_trans")
            train = train.merge(part, how="left",
                                left_on="card_id", right_on="card_id")
            test = test.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()

    with timer("new_trans"):
        for filepath in get_new_trans_downcast():
            part = load_dataframe32(filepath)
            part = rename_columns(part, "new_trans")
            train = train.merge(part, how="left",
                                left_on="card_id", right_on="card_id")
            test = test.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()

    for name in names:
        with timer(name):
            part = load_dataframe32(features_downcast(name))
            part = rename_columns(part, name)
            train = train.merge(part, how="left",
                                left_on="card_id", right_on="card_id")
            test = test.merge(part, how="left",
                              left_on="card_id", right_on="card_id")
            del part
            gc.collect()
    return train, test


def load_train_test_stacking(names):
    with timer("train"):
        train = load_dataframe32(get_train_downcast())
        train = train[["card_id", "target"]].set_index("card_id")

    with timer("test"):
        test = load_dataframe32(get_test_downcast())
        test = test[["card_id"]].set_index("card_id")

    for name in names:
        train[name] = load_oof(name).set_index("card_id")[name]
        test[name] = load_submission(name).set_index("card_id")["target"]
    return train, test


def load_train_test_main_merchant():
    with timer("train"):
        train = load_dataframe32(get_train_downcast())
        train = train[["card_id", "target"]].set_index("card_id")

    with timer("test"):
        test = load_dataframe32(get_test_downcast())
        test = test[["card_id"]].set_index("card_id")

    data = pd.read_csv("./data/train_test_main_merchant.csv")
    features = ["card_id"] + ["main_m_lag{}".format(month) for month in range(16)]
    train = train.merge(data, how="left", left_on="card_id", right_on="card_id")[features + ["target"]]
    test = test.merge(data, how="left", left_on="card_id", right_on="card_id")[features]
    return train, test


def load_train_test_main_merchant_pa():
    with timer("train"):
        train = load_dataframe32(get_train_downcast())
        train = train[["card_id", "target"]].set_index("card_id")

    with timer("test"):
        test = load_dataframe32(get_test_downcast())
        test = test[["card_id"]].set_index("card_id")

    data = pd.read_csv("./data/train_test_main_merchant_pa.csv")
    features = ["card_id"] + ["main_m_lag{}".format(month) for month in range(16)]
    train = train.merge(data, how="left", left_on="card_id", right_on="card_id")[features + ["target"]]
    test = test.merge(data, how="left", left_on="card_id", right_on="card_id")[features]
    return train, test


def save_submission(name, data):
    filepath = gen_submission_filepath(name)
    data.columns = ["card_id", "target"]
    data.to_csv(filepath, index=False)


def load_submission(name):
    return pd.read_csv(gen_submission_filepath(name))


def save_oof(name, data):
    filepath = gen_oof_filepath(name)
    save_dataframe32(filepath, data, keep=["card_id"])


def load_oof(name):
    filepath = gen_oof_filepath(name)
    return load_dataframe32(filepath)


def main():
    # print(get_train_downcast())
    # print(get_test_downcast())
    # print(get_new_trans_downcast())
    # print(get_all_trans_downcast())
    # print(get_hist_trans_downcast())
    # train = load_train_newk_features()
    train = load_test_newk_features()
    print(train.shape)
    for i, (col, type_) in enumerate(zip(train.columns, train.dtypes)):
        print(i, col, type_, train[col].isnull().sum())
    pass


if __name__ == '__main__':
    main()
