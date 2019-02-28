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
import gc
# from data_io import load_train_features, load_test_features
from data_io import load_train_all_features, load_test_all_features
from validator import KFoldValidator
from models import LGBModel
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


def main():
    model_name = "lgb024_dart"
    init_global_logger("train_predict_" + model_name)
    train_data = load_train_all_features()
    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data, encoder_ = encode_feature123(train_data, None, is_train=True)

    train_data = reorder_train_set(train_data)
    print_info("train_data.reorder", train_data.head())

    features = list(train_data.columns)
    features.remove("outlier")
    features.remove("card_id")
    features.remove("target")
    features_discarded = [
        'category_2_3.0_hist_trans',
        'newk_amount_month_ratio_max',
        'newk_duration_max',
        'newk_duration_mean',
        'newk_hist_amount_month_ratio_max',
        'newk_hist_category_3_mean',
        'newk_hist_installments_mean',
        'newk_hist_price_max',
        'newk_hist_price_mean',
        'newk_hist_price_sum',
        'newk_hist_purchase_amount_max',
        'newk_installments_mean',
        'newk_new_category_2_mean',
        'newk_new_category_2_mean_mean',
        'newk_new_duration_mean',
        'newk_new_hour_mean',
        'newk_new_hour_min',
        'newk_new_price_mean',
        'newk_new_purchase_amount_mean',
        'newk_price_max',
        'newk_price_mean',
        'newk_price_total',
        'newk_purchase_amount_max',
        'newk_purchase_amount_mean',
        'purchase_amount_max_hist_trans',
        'purchase_amount_mean_new_trans',
        'purchase_amount_sum_new_trans',
        'ratio_1_lag-4_monthly_merchant_avg_std',
        'ratio_1_lag0_psum',
        'sum_1_lag-4_psum',
        'sum_1_lag1_monthly_pmax',
        'sum_1_lag2_monthly_merchant_pmax',
        'sum_1_lag2_monthly_pmax',
        'sum_2_lag-4_monthly_merchant_pmax',
        'sum_2_lag1_monthly_merchant_pmax',
        'sum_2_lag1_monthly_pmax',
        'sum_2_lag2_psum',
        'sum_3_lag1_monthly_merchant_avg_std',
        'sum_3_lag1_monthly_pmax',
        'sum_3_lag1_psum',
        'sum_3_lag2_monthly_merchant_pmax',
        'sum_3_lag2_monthly_pmax',
        'sum_3_lag2_psum',
        'sum_4_lag1_monthly_merchant_avg_std',
        'sum_4_lag1_monthly_pmax',
        'sum_4_lag1_psum',
        'sum_4_lag2_monthly_merchant_pmax',
        'sum_4_lag2_monthly_pmax',
        'sum_4_lag2_psum'
    ]
    features = [col for col in features if col not in features_discarded]

    for ftr in features:
        print(ftr)


if __name__ == '__main__':
    main()