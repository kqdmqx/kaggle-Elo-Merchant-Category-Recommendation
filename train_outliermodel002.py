import addict
import numpy as np
from data_io import load_train_features, load_test_features
from validator import KFoldValidator
from models import LGBModel
from my_logger import print_info


def main():
    model_name = "outlier002"
    train_data = load_train_features()
    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())

    features = list(train_data.columns)
    features.remove("card_id")
    features.remove("target")

    id_train = train_data.card_id.values
    X_train = train_data[features].values
    y_train = train_data.target.values
    y_train = (y_train < -33).astype(np.int32)

    print_info("id_train.shape", id_train.shape)
    print_info("X_train.shape", X_train.shape)
    print_info("y_train.shape", y_train.shape)

    # test_data = load_test_features()
    # id_test = test_data.card_id.values
    # X_test = test_data[features].values
    # print_info("id_test.shape", id_test.shape)
    # print_info("X_test.shape", X_test.shape)

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
    validator.validate(lgbmodel, name=model_name)
    validator.show_metric()
    # validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
