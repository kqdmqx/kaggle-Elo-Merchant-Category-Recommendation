# 2019-02-19 20:51:46 train_predict_stacking029 >>> [info] metrics.mean : 3.6438722213109336
# 2019-02-19 20:51:46 train_predict_stacking029 >>> [info] metrics.std : 0.012672442186585543

# 2019-02-19 20:53:25 train_predict_stacking029 >>> [info] metrics.mean : 3.643455425898234
# 2019-02-19 20:53:25 train_predict_stacking029 >>> [info] metrics.std : 0.013338493549518608

# 2019-02-21 21:04:14 train_predict_stacking031 >>> [info] metrics.mean : 3.6440197229385376
# 2019-02-21 21:04:14 train_predict_stacking031 >>> [info] metrics.std : 0.01221759474258279

import addict
import pandas as pd
import numpy as np
import gc
# from data_io import load_train_features, load_test_features
# from data_io import load_train_test_all
from data_io import load_train_test_stacking
from validator import KFoldValidator
from models import LRModel
from my_logger import print_info, init_global_logger


def reorder_train_set(train):
    train['rounded_target'] = train['target'].round(0)
    train = train.sort_values('rounded_target').reset_index(drop=True)
    vc = train['rounded_target'].value_counts()
    vc = dict(sorted(vc.items()))
    df = pd.DataFrame()
    train['indexcol'], i = 0, 1
    for k, v in vc.items():
        step = train.shape[0] / v
        indent = train.shape[0] / (v + 1)
        df2 = train[train['rounded_target'] == k].sample(v, random_state=120).reset_index(drop=True)
        for j in range(0, v):
            df2.at[j, 'indexcol'] = indent + j * step + 0.000001 * i
        df = pd.concat([df2, df])
        i += 1

    train = df.sort_values('indexcol', ascending=True).reset_index(drop=True)
    del train['indexcol'], train['rounded_target']
    gc.collect()
    return train


def encode_feature123(data, encoder, is_train=True):
    if is_train:
        data["outlier"] = (data.target < -30)
        encoder = addict.Dict()
    for col in ["feature_1", "feature_2", "feature_3"]:
        if is_train:
            encoder[col] = data.groupby([col])['outlier'].mean()
        data[col] = data[col].map(encoder[col])
    return data, encoder


def recover_sum(data, columns):
    return np.log2(sum([2 ** data[col] for col in columns]) / len(columns))


def main():
    model_name = "stacking031"
    init_global_logger("train_predict_" + model_name)
    names = (
        # "lgb024_dart",
        "lgb025_dart",
        # "lgb026_dart",
        "lgb027",
        "lgb028_gbdt",
        # "lgb028_gbdt_lowlr",
        "xgb028",
        # "rfr030"
    )
    train_data, test_data = load_train_test_stacking(names)
    train_data = train_data.reset_index()
    test_data = test_data.reset_index()

    train_data["rc_mean"] = recover_sum(train_data, names)
    test_data["rc_mean"] = recover_sum(test_data, names)

    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data = reorder_train_set(train_data)
    print_info("train_data.reorder", train_data.head())

    features = list(names) + ["rc_mean"]

    id_train = train_data.card_id.values
    X_train = train_data[features].values
    y_train = train_data.target.values
    print_info("id_train.shape", id_train.shape)
    print_info("X_train.shape", X_train.shape)
    print_info("y_train.shape", y_train.shape)

    id_test = test_data.card_id.values
    X_test = test_data[features].values
    print_info("id_test.shape", id_test.shape)
    print_info("X_test.shape", X_test.shape)

    lr_params = addict.Dict()
    lr_params.metric = "rmse"

    model = LRModel(lr_params)
    print_info("lr_params", model.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test, n_splits=6, shuffle=False)
    validator.validate(model, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
