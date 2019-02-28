# 2019-02-16 02:08:01 train_predict_lgb013 >>> [info] metrics.mean : 3.64638014742162
# 2019-02-16 02:08:01 train_predict_lgb013 >>> [info] metrics.std : 0.09647357244389095

# 2019-02-17 01:09:06 train_predict_lgb014 >>> [info] metrics.mean : 3.6485028360041762
# 2019-02-17 01:09:06 train_predict_lgb014 >>> [info] metrics.std : 0.09924692541508232

# 2019-02-17 01:56:15 train_predict_lgb015 >>> [info] metrics.mean : 3.6447698341661887
# 2019-02-17 01:56:15 train_predict_lgb015 >>> [info] metrics.std : 0.09642637731669208

import addict
import pandas as pd
import optuna
from functools import partial
# from data_io import load_train_features, load_test_features
from data_io import load_train_all_features
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


def target_function(trial, model_name, id_train, X_train, y_train):
    lgb_params = addict.Dict()
    lgb_params.objective = 'regression'
    lgb_params.metric = 'rmse'
    lgb_params.verbosity = -1
    lgb_params.learning_rate = 0.05
    # lgb_params.device = 'gpu'
    lgb_params.num_boost_round = 5000
    lgb_params.early_stopping_rounds = 100
    lgb_params.verbose = -1
    lgb_params.verbose_eval = 50
    lgb_params.seed = 673
    lgb_params.bagging_seed = 673
    lgb_params.drop_seed = 673

    lgb_params.boosting_type = trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss'])
    lgb_params.num_leaves = trial.suggest_int('num_leaves', 16, 64)
    lgb_params.colsample_bytree = trial.suggest_uniform('colsample_bytree', 0.001, 1)
    lgb_params.subsample = trial.suggest_uniform('subsample', 0.001, 1)
    lgb_params.max_depth = trial.suggest_int('max_depth', 5, 20)
    lgb_params.reg_alpha = trial.suggest_uniform('reg_alpha', 0, 10)
    lgb_params.reg_lambda = trial.suggest_uniform('reg_lambda', 0, 10)
    lgb_params.min_split_gain = trial.suggest_uniform('min_split_gain', 0, 10)
    lgb_params.min_child_weight = trial.suggest_uniform('min_child_weight', 0, 45)
    lgb_params.min_data_in_leaf = trial.suggest_int('min_data_in_leaf', 16, 64)

    if lgb_params['boosting_type'] == 'dart':
        lgb_params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        lgb_params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)

    if lgb_params['boosting_type'] == 'goss':
        lgb_params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        lgb_params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - lgb_params['top_rate'])

    lgbmodel = LGBModel(lgb_params)
    # print_info("lgb_params", lgbmodel.model_params)
    validator = KFoldValidator(id_train, X_train, y_train, id_test=None, X_test=None, n_splits=5)
    validator.validate(lgbmodel, name=model_name)
    return validator.get_score(True)


def main():
    model_name = "lgb015"
    init_global_logger("optimize_" + model_name)
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

    objective = partial(target_function,
                        model_name=model_name,
                        id_train=id_train,
                        X_train=X_train,
                        y_train=y_train)

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # save result
    hist_df = study.trials_dataframe()
    hist_df.to_csv("./models/optuna_result_lgbm.csv")

    print_info('optuna LightGBM finished.', hist_df.head())


if __name__ == '__main__':
    main()
