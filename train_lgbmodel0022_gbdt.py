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
    model_name = "lgb021_gbdt"
    init_global_logger("train_predict_" + model_name)
    train_data = load_train_all_features()
    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data, encoder_ = encode_feature123(train_data, None, is_train=True)

    # features = list(train_data.columns)
    # features.remove("outlier")
    # features.remove("card_id")
    # features.remove("target")

    features = list(pd.read_csv("./models/lgb018.csv").feature_name.values)
    features_discarded = [
        'newk_price_max',
        'sum_3_lag2_psum',
        'ratio_1_lag0_psum',
        'purchase_amount_mean_new_trans',
        'category_2_3.0_hist_trans',
        'newk_hist_price_sum',
        'newk_new_hour_mean',
        'newk_new_duration_mean',
        'newk_price_mean',
        'newk_new_hour_min',
        'newk_hist_category_3_mean',
        'purchase_amount_sum_new_trans'
    ]
    features = [col for col in features if col not in features_discarded]

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
    lgb_params.objective = "regression"
    lgb_params.metric = "rmse"
    lgb_params.learning_rate = 0.01  # 0.005

    lgb_params.num_boost_round = 5000
    lgb_params.early_stopping_rounds = 100
    lgb_params.verbose = -1
    lgb_params.verbose_eval = 50
    lgb_params.seed = 673
    lgb_params.bagging_seed = 673
    lgb_params.drop_seed = 673

    lgb_params.boosting_type = 'gbdt'
    lgb_params.colsample_bytree = 0.2490103183630386
    lgb_params.max_depth = 6
    lgb_params.min_child_weight = 15.683032170854897
    lgb_params.min_data_in_leaf = 28
    lgb_params.min_split_gain = 9.979192051521423
    lgb_params.num_leaves = 85
    lgb_params.reg_alpha = 46.46404476987479
    lgb_params.reg_lambda = 45.16884576268798
    lgb_params.subsample = 0.9058813446425535

    lgbmodel = LGBModel(lgb_params)
    print_info("lgb_params", lgbmodel.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test, n_splits=5)
    validator.validate(lgbmodel, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
