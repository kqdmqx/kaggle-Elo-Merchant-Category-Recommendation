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
import numpy as np
import gc
# from data_io import load_train_features, load_test_features
from data_io import load_train_test_all
from validator import KFoldValidator
from models import LGBModel
from my_logger import print_info, init_global_logger
from utils import set_random_seed


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
    set_random_seed(673)
    model_name = "lgb026_dart_null"
    init_global_logger("train_predict_" + model_name)
    train_data, test_data = load_train_test_all()

    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data = reorder_train_set(train_data)
    train_data, encoder_ = encode_feature123(train_data, None, is_train=True)
    test_data, encoder_ = encode_feature123(test_data, encoder_, is_train=False)
    print_info("train_data.reorder", train_data.head())

    # features = list(pd.read_csv("./models/lgb025.csv").feature_name)
    train_dtypes = train_data.dtypes
    features = list(train_dtypes[(train_dtypes == 'float32') | (train_dtypes == 'float64')].index)
    features_discard = [
        'outlier',
        'target',
        'category_2_3.0_hist_trans',
        'newk_amount_month_ratio_max_newk',
        'newk_duration_max_newk',
        'newk_duration_mean_newk',
        'newk_hist_amount_month_ratio_max_newk',
        'newk_hist_category_3_mean_newk',
        'newk_hist_installments_mean_newk',
        'newk_hist_price_max_newk',
        'newk_hist_price_mean_newk',
        'newk_hist_price_sum_newk',
        'newk_hist_purchase_amount_max_newk',
        'newk_installments_mean_newk',
        'newk_new_category_2_mean_mean_newk',
        'newk_new_category_2_mean_newk',
        'newk_new_duration_mean_newk',
        'newk_new_hour_mean_newk',
        'newk_new_hour_min_newk',
        'newk_new_price_mean_newk',
        'newk_new_purchase_amount_mean_newk',
        'newk_price_max_newk',
        'newk_price_mean_newk',
        'newk_price_total_newk',
        'newk_purchase_amount_max_newk',
        'newk_purchase_amount_mean_newk',
        'purchase_amount_max_hist_trans',
        'purchase_amount_mean_new_trans',
        'purchase_amount_sum_new_trans',
        'ratio_1_lag-4_monthly_merchant_avg_std',
        'ratio_1_lag0_monthly_psum',
        'ratio_1_lag12_monthly_merchant_pmax_abs',
        'ratio_1_lag9_monthly_merchant_avg_std_abs',
        'ratio_2_lag12_monthly_pmax_abs',
        'ratio_3_lag12_monthly_pmax_abs',
        'ratio_dura2_duar_count',
        'sum_1_lag-4_monthly_psum',
        'sum_1_lag12_monthly_merchant_avg_std_abs',
        'sum_1_lag13_monthly_merchant_pmax_abs',
        'sum_1_lag13_monthly_pmax_abs',
        'sum_1_lag15_monthly_merchant_pmax_abs',
        'sum_1_lag15_monthly_pmax_abs',
        'sum_1_lag1_monthly_pmax',
        'sum_1_lag2_monthly_merchant_pmax',
        'sum_1_lag2_monthly_pmax',
        'sum_2_lag-4_monthly_merchant_pmax',
        'sum_2_lag15_monthly_merchant_pmax_abs',
        'sum_2_lag15_monthly_pmax_abs',
        'sum_2_lag1_monthly_merchant_pmax',
        'sum_2_lag1_monthly_pmax',
        'sum_2_lag2_monthly_psum',
        'sum_3_lag15_monthly_merchant_pmax_abs',
        'sum_3_lag15_monthly_pmax_abs',
        'sum_3_lag1_monthly_merchant_avg_std',
        'sum_3_lag1_monthly_pmax',
        'sum_3_lag1_monthly_psum',
        'sum_3_lag2_monthly_merchant_pmax',
        'sum_3_lag2_monthly_pmax',
        'sum_3_lag2_monthly_psum',
        'sum_4_lag15_monthly_merchant_pmax_abs',
        'sum_4_lag15_monthly_pmax_abs',
        'sum_4_lag1_monthly_merchant_avg_std',
        'sum_4_lag1_monthly_pmax',
        'sum_4_lag1_monthly_psum',
        'sum_4_lag2_monthly_merchant_pmax',
        'sum_4_lag2_monthly_pmax',
        'sum_4_lag2_monthly_psum'
    ]
    features = [col for col in features if col not in features_discard]

    features_df = pd.DataFrame()
    features_df["feature_name"] = features
    features_df.to_csv("./models/lgb026.csv", index=False)

    id_train = train_data.card_id.values
    X_train = train_data[features].values
    y_train = train_data.target.values
    np.random.shuffle(y_train)
    print_info("id_train.shape", id_train.shape)
    print_info("X_train.shape", X_train.shape)
    print_info("y_train.shape", y_train.shape)

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

    lgb_params.boosting_type = 'dart'
    lgb_params.colsample_bytree = 0.7140058735819552
    lgb_params.drop_rate = 3.921198830433312e-06
    lgb_params.max_depth = 11
    lgb_params.min_child_weight = 28.517727948752466
    lgb_params.min_data_in_leaf = 53
    lgb_params.min_split_gain = 4.660238646763835
    lgb_params.num_leaves = 23
    lgb_params.reg_alpha = 49.03533562166284
    lgb_params.reg_lambda = 36.15997267107921
    lgb_params.skip_drop = 4.0918621947407706e-07
    lgb_params.subsample = 0.47633499039596194

    lgbmodel = LGBModel(lgb_params)
    print_info("lgb_params", lgbmodel.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test, n_splits=6, shuffle=False)
    validator.validate(lgbmodel, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
