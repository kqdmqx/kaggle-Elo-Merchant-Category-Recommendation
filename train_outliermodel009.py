import addict
import numpy as np
from data_io import load_train_features, load_test_features
from validator import KFoldValidator
from models import LGBModel
from my_logger import print_info, init_global_logger


def merge_feature(data, encoder, is_train):
    data["feature_mg"] = (data.feature_1 * 100 + data.feature_2 * 10 + data.feature_3).astype(np.int32)
    if is_train:
        data["outlier"] = (data.target < -33).astype(np.int32)
        encoder.freq_encoder = data.feature_mg.value_counts()
        encoder.outlier_encoder = data.groupby("feature_mg").outlier.mean()
        encoder.raw_encoder = data.groupby("feature_mg").target.mean()
        encoder.pure_encoder = data[data.target > -33].groupby("feature_mg").target.mean()
    data["feature_freq"] = data.feature_mg.map(encoder.freq_encoder)
    data["feature_outlier"] = data.feature_mg.map(encoder.outlier_encoder)
    data["feature_raw"] = data.feature_mg.map(encoder.raw_encoder)
    data["feature_pure"] = data.feature_mg.map(encoder.pure_encoder)
    return data


def main():
    model_name = "outlier-009"
    init_global_logger("train_predict_" + model_name)
    train_data = load_train_features()
    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())

    encoder = addict.Dict()
    train_data = merge_feature(train_data, encoder, is_train=True)

    features = list(train_data.columns)
    features.remove("card_id")
    features.remove("target")
    features.remove("outlier")
    features.remove("feature_1")
    features.remove("feature_2")
    features.remove("feature_3")
    features += [
        "feature_freq",
        "feature_outlier",
        "feature_raw",
        "feature_pure"
    ]

    id_train = train_data.card_id.values
    X_train = train_data[features].values
    y_train = train_data.target.values
    y_train = (y_train < -33).astype(np.int32)

    print_info("id_train.shape", id_train.shape)
    print_info("X_train.shape", X_train.shape)
    print_info("y_train.shape", y_train.shape)

    test_data = load_test_features()
    test_data = merge_feature(test_data, encoder, is_train=False)

    id_test = test_data.card_id.values
    X_test = test_data[features].values
    print_info("id_test.shape", id_test.shape)
    print_info("X_test.shape", X_test.shape)

    lgb_params = addict.Dict()
    lgb_params.boosting_type = "gbdt"
    lgb_params.objective = "binary"
    lgb_params.metric = "auc"
    lgb_params.learning_rate = 0.005  # 0.005
    lgb_params.max_bin = 320
    lgb_params.max_depth = -1
    lgb_params.num_leaves = 541
    lgb_params.min_child_samples = 324
    lgb_params.subsample = 0.9
    lgb_params.subsample_freq = 1
    lgb_params.colsample_bytree = 0.792
    lgb_params.min_gain_to_split = 0.533
    lgb_params.reg_lambda = 68.658
    lgb_params.reg_alpha = 27.682
    lgb_params.is_unbalance = True
    lgb_params.num_boost_round = 5000
    lgb_params.early_stopping_rounds = 100
    lgb_params.verbose = -1
    lgb_params.verbose_eval = 50

    lgbmodel = LGBModel(lgb_params)
    print_info("lgb_params", lgbmodel.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, None, None)
    validator.show_folds()
    # validator.set_keep_out(None)
    validator.validate(lgbmodel, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
