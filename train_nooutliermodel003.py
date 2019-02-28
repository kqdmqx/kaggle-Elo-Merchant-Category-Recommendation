# [info] metrics.mean : 3.655208149559866
# [info] metrics.std : 0.10312237648745122

import addict
from data_io import load_train_features, load_test_features
from validator import KFoldValidator
from models import LGBModel
from my_logger import print_info, init_global_logger


def main():
    model_name = "no-outlier-003"
    init_global_logger("train_predict_" + model_name)
    train_data = load_train_features()
    train_data = train_data[train_data.target > -33]
    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())

    features = list(train_data.columns)
    features.remove("card_id")
    features.remove("target")

    id_train = train_data.card_id.values
    X_train = train_data[features].values
    y_train = train_data.target.values

    print_info("id_train.shape", id_train.shape)
    print_info("X_train.shape", X_train.shape)
    print_info("y_train.shape", y_train.shape)

    test_data = load_test_features()
    id_test = test_data.card_id.values
    X_test = test_data[features].values
    print_info("id_test.shape", id_test.shape)
    print_info("X_test.shape", X_test.shape)

    lgb_params = addict.Dict()
    lgb_params.boosting_type = "gbdt"
    lgb_params.objective = "regression"
    lgb_params.metric = "rmse"
    lgb_params.learning_rate = 0.005  # 0.005
    lgb_params.max_bin = 414
    lgb_params.max_depth = 22
    lgb_params.num_leaves = 575
    lgb_params.min_child_samples = 158
    lgb_params.subsample = 0.7235252307967774
    lgb_params.subsample_freq = 3
    lgb_params.colsample_bytree = 0.3795114750551951
    lgb_params.min_gain_to_split = 15.744206187155678
    lgb_params.reg_lambda = 127.13746592972745
    lgb_params.reg_alpha = 155.9021468946609
    lgb_params.is_unbalance = True
    lgb_params.num_boost_round = 5000
    lgb_params.early_stopping_rounds = 100
    lgb_params.verbose = -1
    lgb_params.verbose_eval = 50

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
    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test)
    validator.validate(lgbmodel, name=model_name)
    validator.show_metric()
    validator.predict_oof(name=model_name)
    validator.predict_test(name=model_name)


if __name__ == '__main__':
    main()
