import glob
import gc
import pandas as pd
import numpy as np
from downcast import load_dataframe32


def get_train_downcast():
    return "./data/features_downcast/train"


def get_test_downcast():
    return "./data/features_downcast/test"


def get_hist_trans_downcast():
    lines = glob.glob("./data/features_downcast/historical_transactions*.data64.npy")
    return [line.replace(".data64.npy", "").replace("\\", "/") for line in lines]


def get_new_trans_downcast():
    lines = glob.glob("./data/features_downcast/new_merchant_transactions*.data64.npy")
    return [line.replace(".data64.npy", "").replace("\\", "/") for line in lines]


def get_all_trans_downcast():
    lines = glob.glob("./data/features_downcast/all_transactions_transactions*.data64.npy")
    return [line.replace(".data64.npy", "").replace("\\", "/") for line in lines]


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


def load_train_features():
    train = load_dataframe32(get_train_downcast())
    train = append_time_features(train)
    for filepath in get_all_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "all_trans")
        train = train.merge(part, how="left", left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(train.shape)
    for filepath in get_hist_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "hist_trans")
        train = train.merge(part, how="left", left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(train.shape)
    for filepath in get_new_trans_downcast():
        part = load_dataframe32(filepath)
        part = rename_columns(part, "new_trans")
        train = train.merge(part, how="left", left_on="card_id", right_on="card_id")
        del part
        gc.collect()
    # print(train.shape)
    return train


def main():
    # print(get_train_downcast())
    # print(get_test_downcast())
    # print(get_new_trans_downcast())
    # print(get_all_trans_downcast())
    # print(get_hist_trans_downcast())
    train = load_train_features()
    print(train.shape)
    for i, (col, type_) in enumerate(zip(train.columns, train.dtypes)):
        print(i, col, type_)
    pass


if __name__ == '__main__':
    main()
