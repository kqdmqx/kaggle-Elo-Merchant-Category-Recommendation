import addict
from data_io import load_train_features
from optimizer import adapt_optimizer
from my_logger import print_info, set_global_slience_flag


def main():
    train_data = load_train_features()
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

    # test_data = load_test_features()
    # id_test = test_data.card_id.values
    # X_test = test_data[features].values
    # print_info("id_test.shape", id_test.shape)
    # print_info("X_test.shape", X_test.shape)

    lgb_space = addict.Dict()
    lgb_space.max_bin = [50, 500]
    lgb_space.max_depth = [3, 25]
    lgb_space.num_leaves = [100, 600]
    lgb_space.min_child_samples = [10, 500]
    lgb_space.subsample = [0.25, 0.95]
    lgb_space.subsample_freq = [1, 4]
    lgb_space.colsample_bytree = [0.25, 0.95]
    lgb_space.min_gain_to_split = [0.01, 50.0]
    lgb_space.reg_lambda = [0., 200.]
    lgb_space.reg_alpha = [0., 200.]

    opt_config = addict.Dict()
    opt_config.random_seed = 67373
    opt_config.space = lgb_space

    opt = adapt_optimizer("lgb", opt_config)
    opt.set_data(X_train, y_train)
    set_global_slience_flag(True)
    opt.maxmize(10, 30)
    set_global_slience_flag(False)
    opt.show()
    opt.save("optimizer-lgbmodel000")


if __name__ == '__main__':
    main()
