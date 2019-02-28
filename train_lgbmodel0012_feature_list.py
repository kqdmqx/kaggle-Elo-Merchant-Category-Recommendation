# 2019-02-13 18:33:02 train_predict_lgb011 >>> [info] metrics.mean : 3.6438485970313406
# 2019-02-13 18:33:02 train_predict_lgb011 >>> [info] metrics.std : 0.09697449778532569

import addict
import pandas as pd
# from data_io import load_train_features, load_test_features
from data_io import load_train_psum_features, load_test_psum_features
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
    model_name = "lgb012"
    init_global_logger("train_predict_" + model_name)
    train_data = load_train_psum_features()
    print_info("train_data.shape", train_data.shape)
    print_info("train_data.head", train_data.head())
    train_data, encoder_ = encode_feature123(train_data, None, is_train=True)

    # features = list(train_data.columns)
    # features.remove("outlier")
    # features.remove("card_id")
    # features.remove("target")

    features = list(pd.read_csv("./models/lgb011.csv").feature_name.values)
    features_psum = [col for col in train_data.columns if col.endswith("psum")]
    features += features_psum

    for ftr in features:
        print(ftr)


if __name__ == '__main__':
    main()
