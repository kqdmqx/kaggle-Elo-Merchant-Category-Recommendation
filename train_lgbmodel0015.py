# 2019-02-16 02:08:01 train_predict_lgb013 >>> [info] metrics.mean : 3.64638014742162
# 2019-02-16 02:08:01 train_predict_lgb013 >>> [info] metrics.std : 0.09647357244389095

# 2019-02-17 01:09:06 train_predict_lgb014 >>> [info] metrics.mean : 3.6485028360041762
# 2019-02-17 01:09:06 train_predict_lgb014 >>> [info] metrics.std : 0.09924692541508232

# 2019-02-17 01:56:15 train_predict_lgb015 >>> [info] metrics.mean : 3.6447698341661887
# 2019-02-17 01:56:15 train_predict_lgb015 >>> [info] metrics.std : 0.09642637731669208

import addict
import pandas as pd
# from data_io import load_train_features, load_test_features
from data_io import load_train_all_features, load_test_all_features
from validator import KFoldValidator
from models import LGBModel
from my_logger import print_info, init_global_logger


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
    model_name = "lgb015"
    init_global_logger("train_predict_" + model_name)
    train_data = load_train_all_features()
    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data, encoder_ = encode_feature123(train_data, None, is_train=True)

    # features = list(train_data.columns)
    # features.remove("outlier")
    # features.remove("card_id")
    # features.remove("target")

    features = list(pd.read_csv("./models/lgb015.csv").feature_name.values)

    id_train = train_data.card_id.values
    X_train = train_data[features].values
    y_train = train_data.target.values

    print_info("id_train.shape", id_train.shape)
    print_info("X_train.shape", X_train.shape)
    print_info("y_train.shape", y_train.shape)

    test_data = load_test_all_features()
    test_data, encoder_ = encode_feature123(test_data, encoder_, is_train=False)

    id_test = test_data.card_id.values
    X_test = test_data[features].values
    print_info("id_test.shape", id_test.shape)
    print_info("X_test.shape", X_test.shape)

    lgb_params = addict.Dict()
    lgb_params.boosting_type = "gbdt"  # "goss"
    lgb_params.objective = "regression"
    lgb_params.metric = "rmse"
    lgb_params.learning_rate = 0.01  # 0.005
    lgb_params.max_bin = 414
    lgb_params.max_depth = 7
    lgb_params.num_leaves = 63
    # lgb_params.min_child_samples = 41
    lgb_params.min_child_weight = 41.9612869171337
    lgb_params.other_rate = .0721768246018207
    lgb_params.subsample = 0.9855232997390695
    lgb_params.subsample_freq = 3
    lgb_params.top_rate = .9064148448434349
    lgb_params.colsample_bytree = 0.5665320670155495
    lgb_params.min_gain_to_split = 9.820197773625843
    lgb_params.min_data_in_leaf = 21
    lgb_params.reg_lambda = 8.2532317400459
    lgb_params.reg_alpha = 9.677537745007898
    lgb_params.is_unbalance = True
    lgb_params.num_boost_round = 5000
    lgb_params.early_stopping_rounds = 100
    lgb_params.verbose = -1
    lgb_params.verbose_eval = 50
    lgb_params.seed = 673
    lgb_params.bagging_seed = 673
    lgb_params.drop_seed = 673

    {
        'target': -3.65681153954509,
        'params': {
            'colsample_bytree': 0.3795114750551951,
            'max_bin': 414.14556353662124,
            'max_depth': 22.627508836815423,
            'min_child_samples': 158.81147323903534,
            'min_gain_to_split': 15.744206187155678,
            'num_leaves': 575.0872370911022,
            'reg_alpha': 155.9021468946609,
            'reg_lambda': 127.13746592972745,
            'subsample': 0.7235252307967774,
            'subsample_freq': 3.500725207911061
        }
    }

    lgbmodel = LGBModel(lgb_params)
    print_info("lgb_params", lgbmodel.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test, n_splits=11)
    validator.validate(lgbmodel, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
