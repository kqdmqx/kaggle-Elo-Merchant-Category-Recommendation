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
import gc
# from data_io import load_train_features, load_test_features
from data_io import load_train_test_all
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
    model_name = "lgb035_gbdt"
    init_global_logger("train_predict_" + model_name)
    names = ("newk",
             "monthly_psum",
             "monthly_pmax",
             "monthly_merchant_pmax",
             "main_merchant_count",
             "monthly_merchant_avg_std",
             "monthly_pmax_abs",
             "monthly_merchant_pmax_abs",
             "duar_count",
             "monthly_merchant_avg_std_abs",
             "nmf100",
             "fastica50",
             "kernelpca")
    train_data, test_data = load_train_test_all(names=names)

    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data = reorder_train_set(train_data)
    train_data, encoder_ = encode_feature123(train_data, None, is_train=True)
    test_data, encoder_ = encode_feature123(test_data, encoder_, is_train=False)
    print_info("train_data.reorder", train_data.head())

    features_df = pd.read_csv("./models/lgb034_fi.csv")
    features = list(features_df[features_df.fi_complex > 5.579243149147489].feature_name)
    # features += [col for col in train_data.columns if col.endswith("nmf100")]
    # features += [col for col in train_data.columns if col.endswith("fastica50")]
    # features += [col for col in train_data.columns if col.endswith("kernelpca")]
    pd.DataFrame({"feature_name": features}).to_csv("./models/lgb035.csv", index=False)

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
    lgb_params.num_leaves = 125
    lgb_params.colsample_bytree = 0.19080406903424613
    lgb_params.subsample = 0.46109355997337415
    lgb_params.max_depth = 8
    lgb_params.reg_alpha = 237.60042941954896
    lgb_params.reg_lambda = 19.607814182782022
    lgb_params.min_split_gain = 3.382411616843602
    lgb_params.min_child_weight = 32.401827720051536
    lgb_params.min_data_in_leaf = 31

    lgbmodel = LGBModel(lgb_params)
    print_info("lgb_params", lgbmodel.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test, n_splits=6, shuffle=False)
    validator.validate(lgbmodel, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
