# 2019-02-16 02:08:01 train_predict_lgb013 >>> [info] metrics.mean : 3.64638014742162
# 2019-02-16 02:08:01 train_predict_lgb013 >>> [info] metrics.std : 0.09647357244389095

# 2019-02-17 01:09:06 train_predict_lgb014 >>> [info] metrics.mean : 3.6485028360041762
# 2019-02-17 01:09:06 train_predict_lgb014 >>> [info] metrics.std : 0.09924692541508232

# 2019-02-17 01:56:15 train_predict_lgb015 >>> [info] metrics.mean : 3.6447698341661887
# 2019-02-17 01:56:15 train_predict_lgb015 >>> [info] metrics.std : 0.09642637731669208

# 2019-02-17 18:58:51 train_predict_lgb016_dart >>> [info] metrics.mean : 3.6423456904931584
# 2019-02-17 18:58:51 train_predict_lgb016_dart >>> [info] metrics.std : 0.0981386805653079

# 2019-02-17 20:41:57 train_predict_lgb017_dart >>> [info] metrics.mean : 3.64380296456501
# 2019-02-17 20:41:57 train_predict_lgb017_dart >>> [info] metrics.std : 0.09880749789041213

# 2019-02-17 21:19:09 train_predict_lgb018_dart >>> [info] metrics.mean : 3.6443214515023583
# 2019-02-17 21:19:09 train_predict_lgb018_dart >>> [info] metrics.std : 0.09706774365319155

# 2019-02-17 21:40:12 train_predict_lgb018_goss >>> [info] metrics.mean : 3.644246859819636
# 2019-02-17 21:40:12 train_predict_lgb018_goss >>> [info] metrics.std : 0.09903328067542437

# 2019-02-17 23:16:31 train_predict_lgb020_goss >>> [info] metrics.mean : 3.6446935688098723
# 2019-02-17 23:16:31 train_predict_lgb020_goss >>> [info] metrics.std : 0.09871755821589523

# 2019-02-19 10:33:04 train_predict_lgb028_gbdt >>> [info] metrics.mean : 3.6452848375535374
# 2019-02-19 10:33:04 train_predict_lgb028_gbdt >>> [info] metrics.std : 0.012914940014618315

# 2019-02-22 23:03:57 train_predict_lgb031_gbdt >>> [info] metrics.mean : 3.6440578978911202
# 2019-02-22 23:03:57 train_predict_lgb031_gbdt >>> [info] metrics.std : 0.013375130692768447

# 2019-02-23 09:57:36 train_predict_lgb034_gbdt >>> [info] metrics.mean : 3.643288840626988
# 2019-02-23 09:57:36 train_predict_lgb034_gbdt >>> [info] metrics.std : 0.012244869501138832

# 2019-02-23 15:14:11 train_predict_lgb035_gbdt >>> [info] metrics.mean : 3.6386944979341855
# 2019-02-23 15:14:11 train_predict_lgb035_gbdt >>> [info] metrics.std : 0.012070215322948764

import addict
import pandas as pd
import numpy as np
import gc
# from data_io import load_train_features, load_test_features
from data_io import load_train_test_all, load_train, load_test
from validator import KFoldValidator
from models import LGBModel
from models_keras2 import KerasModel
from downcast import load_dataframe32
from my_logger import print_info, init_global_logger
from filepath_collection import features_downcast
from data_io import rename_columns


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


def main():
    model_name = "keras043"
    init_global_logger("train_predict_" + model_name)
    train_data = load_train()
    test_data = load_test()

    names = ("main_m_code16",
             "main_m_value16",
             "topk_m_code150",
             "topk_m_value150")

    for name in names:
        part = load_dataframe32(features_downcast(name))
        print(name)
        print(part.shape)
        part = rename_columns(part, name)
        print(part.shape)
        train_data = train_data.merge(part, how="left", left_on="card_id", right_on="card_id")
        test_data = test_data.merge(part, how="left", left_on="card_id", right_on="card_id")

    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data = reorder_train_set(train_data)
    print_info("train_data.reorder", train_data.head())

    def get_columns(data, name):
        return [col for col in data.columns if col.endswith(name)]

    id_train = train_data.card_id.values
    X_train = {
        name: train_data[get_columns(train_data, name)].values
        for name in names
    }
    y_train = train_data.target.values
    print_info("id_train.shape", id_train.shape)
    # print_info("X_train.shape", X_train.shape)
    print_info("y_train.shape", y_train.shape)
    id_test = test_data.card_id.values
    X_test = {
        name: test_data[get_columns(test_data, name)].values
        for name in names
    }
    print_info("id_test.shape", id_test.shape)
    # print_info("X_test.shape", X_test.shape)

    # X_train[np.isnan(X_train)] = 0.0
    # X_test[np.isnan(X_test)] = 0.0

    # max_value = 10000
    # min_value = -10000
    # print_info("X_train.max", max_value)
    # print_info("X_train.min", min_value)

    # X_train[X_train > max_value] = max_value
    # X_train[X_train < min_value] = min_value

    # X_test[X_test > max_value] = max_value
    # X_test[X_test < min_value] = min_value

    keras_params = addict.Dict()
    keras_params.model_name = model_name
    keras_params.metric = "rmse"
    keras_params.embed_size = 128  # how big is each word vector
    keras_params.max_features = 244022  # how many unique words to use (i.e num rows in embedding vector)
    keras_params.maxlen = 16  # max number of words in a question to use
    keras_params.batch_size = 64
    keras_params.epochs = 200
    keras_params.lr = 0.1

    kerasmodel = KerasModel(keras_params)
    print_info("keras_params", kerasmodel.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test, n_splits=6, shuffle=False)
    validator.validate(kerasmodel, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
