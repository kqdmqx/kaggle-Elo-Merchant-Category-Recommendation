import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import KFold
import time
import sys
import datetime
import gc

from downcast import save_dataframe32, load_dataframe32
from filepath_collection import features_downcast
from my_logger import print_info, init_global_logger

import addict


def print_mem(data):
    gb = round(sys.getsizeof(data) / 1024 / 1024 / 1024, 2)
    print("{} GB".format(gb))
    return gb


def keep_cat_part(data):
    del data["merchant_id"]
    del data["purchase_amount"]
    del data["purchase_date"]
    gc.collect()
    return data


def calc_yearmonth(series):
    return series.dt.year * 100 + series.dt.month


def append_time_features(data):
    data["purchase_date"] = pd.to_datetime(data.purchase_date)
    data["purchase_month"] = calc_yearmonth(data.purchase_date)
    data["purchase_hour"] = data.purchase_date.dt.hour
    return data


def calc_cat_cnt(data, col, key="card_id"):
    nodup = data.drop_duplicates(subset=[key, col])
    cnt = nodup.groupby(key).size()
    return cnt


COLUMNS_TRANSACTION_CAT = [
    'authorized_flag',
    'city_id',
    'category_1',
    'installments',
    'category_3',
    'merchant_id',
    'merchant_category_id',
    'month_lag',
    'category_2',
    'state_id',
    'subsector_id',
    'purchase_month',
    'purchase_hour'
]


def get_cnt_features(data, columns=COLUMNS_TRANSACTION_CAT):
    data_cat_cnt = addict.Dict()
    for col in columns:
        tar_col = col + "_classes"
        data_cat_cnt[tar_col] = calc_cat_cnt(data, col)
        print_info("count classes", "{}->{}".format(col, tar_col))
        gc.collect()
    result = pd.DataFrame(data_cat_cnt)
    result.reset_index(inplace=True)
    return result


def get_cat_frequent(data, col, key="card_id"):
    dummies = pd.get_dummies(data[col])
    dummies["card_id"] = data[key]
    result = dummies.groupby(key).sum()
    result.rename(columns=lambda x: "{}_{}".format(col, x), inplace=True)
    result.reset_index(inplace=True)
    return result


def clean_long_tail(data, col, topk=40, default=67373):
    top_set = set(data[col].value_counts().head(topk).index)
    data.loc[~data[col].isin(top_set), col] = default
    return data


COLUMNS_FREQUENT_CAT = [
    'authorized_flag',
    'city_id',
    'category_1',
    'installments',
    'category_3',
    'merchant_category_id',
    'month_lag',
    'category_2',
    'state_id',
    'subsector_id',
    'purchase_month',
    'purchase_hour'
]


def process_cat_features(data, header, columns=COLUMNS_FREQUENT_CAT):
    data = append_time_features(data)
    result_cat_cnt = get_cnt_features(data)
    filepath = features_downcast("{}_{}".format(header, "classes"))
    save_dataframe32(filepath, result_cat_cnt, keep=["card_id"])
    data = clean_long_tail(data, "city_id")
    data = clean_long_tail(data, "merchant_category_id")
    data = keep_cat_part(data)
    for col in columns:
        frequent = get_cat_frequent(data, col)
        filepath = features_downcast("{}_{}_{}".format(header, "frequent", col))
        save_dataframe32(filepath, frequent, keep=["card_id"])
        print_info("calc frequent", col)
        del frequent
        gc.collect()


def process_transaction_cat():
    init_global_logger("preprocessing_transaction_cat")

    history_transactions = pd.read_csv("./data/historical_transactions.csv")
    print_info("load history_transactions", history_transactions.shape)
    process_cat_features(history_transactions, "historical_transactions")
    del history_transactions
    gc.collect()

    new_transactions = pd.read_csv("./data/new_merchant_transactions.csv")
    print_info("load new_transactions", new_transactions.shape)
    process_cat_features(new_transactions, "new_merchant_transactions")
    del new_transactions
    gc.collect()

    history_transactions = pd.read_csv("./data/historical_transactions.csv")
    new_transactions = pd.read_csv("./data/new_merchant_transactions.csv")
    all_transactions = pd.concat([history_transactions, new_transactions])
    print_info("load all_transactions", all_transactions.shape)
    del history_transactions, new_transactions
    gc.collect()
    process_cat_features(all_transactions, "all_transactions")


def get_num_agg(data):
    agg_func = {
        'purchase_amount': ['sum', 'mean', 'max', 'min', 'std'],
        'installments': ['sum', 'mean', 'max', 'min', 'std'],
        'purchase_month': ["min", "max"],
        'month_lag': ['min', 'max']
    }
    result = data.groupby(['card_id']).agg(agg_func)
    result.columns = ['_'.join(col).strip() for col in result.columns.values]
    result.reset_index(inplace=True)
    return result


def process_transaction_num():
    init_global_logger("preprocessing_transaction_num")

    history_transactions = pd.read_csv("./data/historical_transactions.csv")
    print_info("load history_transactions", history_transactions.shape)
    history_transactions = append_time_features(history_transactions)
    history_agg = get_num_agg(history_transactions)
    # print(history_agg.head())
    # return
    filepath = features_downcast("{}_{}".format("historical_transactions", "agg"))
    save_dataframe32(filepath, history_agg, keep=["card_id"])
    del history_transactions, history_agg
    gc.collect()

    new_transactions = pd.read_csv("./data/new_merchant_transactions.csv")
    print_info("load new_transactions", new_transactions.shape)
    new_transactions = append_time_features(new_transactions)
    new_agg = get_num_agg(new_transactions)
    filepath = features_downcast("{}_{}".format("new_merchant_transactions", "agg"))
    save_dataframe32(filepath, new_agg, keep=["card_id"])
    del new_transactions, new_agg
    gc.collect()


def downcast_train_test():
    train = pd.read_csv("./data/train.csv")
    print(train.shape, train.card_id.nunique())
    filepath = features_downcast("train")
    save_dataframe32(filepath, train, keep=["card_id", "first_active_month"])
    test = pd.read_csv("./data/test.csv")
    print(test.shape, test.card_id.nunique())
    filepath = features_downcast("test")
    save_dataframe32(filepath, test, keep=["card_id", "first_active_month"])


def main():
    # process_transaction_cat()
    # process_transaction_num()
    downcast_train_test()
    pass


if __name__ == '__main__':
    main()
