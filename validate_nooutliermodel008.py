# [info] metrics.mean : 3.655208149559866
# [info] metrics.std : 0.10312237648745122

import numpy as np
from data_io import load_train_features
from data_io import load_test_features
from data_io import load_oof
from validator import KFoldValidator
from my_logger import print_info, init_global_logger
from sklearn.metrics import mean_squared_error


def rmse(t, p):
    return np.sqrt(mean_squared_error(t, p))


def main():
    model_name = "no-outlier-006"
    init_global_logger("validate_" + model_name)
    train_data = load_train_features()
    # train_data = train_data[(train_data.target > -33) & (train_data.target != 0)]
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

    validator = KFoldValidator(id_train, X_train, y_train, id_test, X_test)
    keep_out_id = np.where(y_train < -33)[0]
    validator.set_keep_out(keep_out_id)

    model_name = "no-outlier-006"
    print_info("valiate_oof.keep_out", model_name)
    validator.validate_oof(name=model_name, metric_func=rmse, use_keep_out=True)
    validator.show_metric()

    print_info("valiate_oof.not_keep_out", model_name)
    validator.validate_oof(name=model_name, metric_func=rmse, use_keep_out=False)
    validator.show_metric()

    model_name = "lgb007"
    print_info("valiate_oof.keep_out", model_name)
    validator.validate_oof(name=model_name, metric_func=rmse, use_keep_out=True)
    validator.show_metric()

    print_info("valiate_oof.not_keep_out", model_name)
    validator.validate_oof(name=model_name, metric_func=rmse, use_keep_out=False)
    validator.show_metric()

    # mix
    oof_no = load_oof("no-outlier-006")
    oof_all = load_oof("lgb007")
    oof_o = load_oof("outlier002")

    # print(oof_no.head())
    # print(oof_all.head())
    # print(oof_o.head())
    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 > 0.82
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("valiate_oof.keep_out", "mix")
    validator.validate_oof_df(oof_mix, metric_func=rmse, use_keep_out=True)
    validator.show_metric()

    print_info("valiate_oof.not_keep_out", "mix")
    validator.validate_oof_df(oof_mix, metric_func=rmse, use_keep_out=False)
    validator.show_metric()

    print_info("--oof--", '--no-outlier--')
    print_info("full", rmse(oof_no.target, oof_no["no-outlier-006"]))
    print_info("clean", rmse(oof_no[oof_no.target > -33].target, oof_no[oof_no.target > -33]["no-outlier-006"]))

    print_info("--oof--", '--outlier--')
    print_info("full", rmse(oof_all.target, oof_all["lgb007"]))
    print_info("clean", rmse(oof_all[oof_all.target > -33].target, oof_all[oof_all.target > -33]["lgb007"]))

    print_info("--oof--", '--mix.82--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))

    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 > 0.95
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("--oof--", '--mix.95--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))

    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 > 0.75
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("--oof--", '--mix.75--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))

    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 > 0.55
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("--oof--", '--mix.55--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))

    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 > 0.25
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("--oof--", '--mix.25--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))

    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 > 0.2
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("--oof--", '--mix.2--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))

    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 < 0.25
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("--oof--", '--mix.neg.25--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))

    oof_mix = oof_no.copy()
    oof_mix["oof"] = oof_mix["no-outlier-006"]
    maybe = oof_o.outlier002 < 0.45
    oof_mix.loc[maybe, "oof"] = oof_all.loc[maybe, "lgb007"]

    print_info("--oof--", '--mix.neg.45--')
    print_info("full", rmse(oof_mix.target, oof_mix["oof"]))
    print_info("clean", rmse(oof_mix[oof_mix.target > -33].target, oof_mix[oof_mix.target > -33]["oof"]))


if __name__ == '__main__':
    main()
