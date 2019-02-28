import numpy as np
import pandas as pd
from filepath_collection import gen_model_filepath
from my_logger import print_info
# from sklearn.model_selection import GroupKFold
from sklearn.model_selection import KFold
from data_io import save_submission, save_oof, load_oof
from utils import load_obj


def enumerate_ids(generator):
    return [(train_id, valid_id) for train_id, valid_id in generator]


def make_kfold(X, y, n_splits, seed, shuffle=True):
    # kfold = KFold(n_splits, shuffle=True, random_state=seed)
    kfold = KFold(n_splits, shuffle=shuffle, random_state=seed)
    return enumerate_ids(kfold.split(X))


def filter_idx(src_arr, keep_out):
    if keep_out is None:
        return src_arr
    return np.array(list(set(src_arr) - set(keep_out)))


def get_id_from_dict(data, trn):
    return {key: data[key][trn] for key in data}


def get_id_from_data(data, trn):
    if isinstance(data, np.ndarray):
        return data[trn]
    if isinstance(data, dict):
        return get_id_from_dict(data, trn)
    return data[trn]


def get_shape(data):
    if isinstance(data, np.ndarray):
        return data.shape
    if isinstance(data, dict):
        return {key: data[key].shape for key in data}
    return data.shape


class KFoldValidator:
    def __init__(self,
                 id_train,
                 X_train,
                 y_train,
                 id_test,
                 X_test,
                 n_splits=5,
                 how="kfold",
                 seed=67373,
                 shuffle=True):
        self.id_train = id_train
        self.X_train = X_train
        self.y_train = y_train
        self.id_test = id_test
        self.X_test = X_test
        self.fold_ids = make_kfold(y_train, y_train, n_splits, seed, shuffle)
        self.keep_out = None
        self.metrics = []

    def set_keep_out(self, keep_out):
        self.keep_out = None if keep_out is None else set(keep_out)

    def validate(self, model, name, save_model=True):
        self.metrics = []
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            trn = filter_idx(trn, self.keep_out)
            val = filter_idx(val, self.keep_out)
            # print_info("trn fold_{}.shape".format(fold_id), trn.shape)
            # print_info("trn fold_{}.min".format(fold_id), trn.min())
            # print_info("trn fold_{}.max".format(fold_id), trn.max())
            # print_info("val fold_{}.shape".format(fold_id), val.shape)
            # print_info("val fold_{}.min".format(fold_id), val.min())
            # print_info("val fold_{}.max".format(fold_id), val.max())
            X_trn, y_trn = get_id_from_data(self.X_train, trn), self.y_train[trn]
            X_val, y_val = get_id_from_data(self.X_train, val), self.y_train[val]
            model.train(X_trn, y_trn, X_val, y_val)
            valid_score = model.get_valid_score()
            self.metrics.append(valid_score)
            filepath = gen_model_filepath("{}_{}".format(name, fold_id))
            if save_model:
                model.save(filepath)

    def validate_oof(self, name, metric_func, use_keep_out=False):
        self.metrics = []
        oof_df = load_oof(name)
        oof_target = oof_df[name].values
        oof_real = oof_df.target.values
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            if use_keep_out:
                val = filter_idx(val, self.keep_out)
            print_info("ford {}.shape".format(fold_id), len(val))
            # y_val = self.y_train[val]
            y_val = oof_real[val]
            pred_val = oof_target[val]
            # print_info("pred_val.shape", pred_val.shape)
            # print_info("pred_val.mean", pred_val.mean())
            # print_info("pred_val.std", pred_val.std())
            # print_info("y_val.shape", y_val.shape)
            # print_info("y_val.mean", y_val.mean())
            self.metrics.append(metric_func(y_val, pred_val))
        print_info("metrics", self.metrics)

    def validate_oof_df(self, oof_df, metric_func, use_keep_out=False):
        self.metrics = []
        oof_target = oof_df["oof"].values
        oof_real = oof_df.target.values
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            if use_keep_out:
                val = filter_idx(val, self.keep_out)
            print_info("ford {}.shape".format(fold_id), len(val))
            # y_val = self.y_train[val]
            y_val = oof_real[val]
            pred_val = oof_target[val]
            # print_info("pred_val.shape", pred_val.shape)
            # print_info("pred_val.mean", pred_val.mean())
            # print_info("pred_val.std", pred_val.std())
            # print_info("y_val.shape", y_val.shape)
            # print_info("y_val.mean", y_val.mean())
            self.metrics.append(metric_func(y_val, pred_val))
        print_info("metrics", self.metrics)

    def predict_oof(self, name):
        oof = np.zeros(self.y_train.shape)
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            X_val = get_id_from_data(self.X_train, val)
            filepath = gen_model_filepath("{}_{}".format(name, fold_id))
            model = load_obj(filepath)
            oof[val] = model.predict(X_val)
        oof_df = pd.DataFrame({"card_id": self.id_train, name: oof, "target": self.y_train})
        save_oof(name, oof_df)
        return oof_df

    def predict_test(self, name):
        predictions = []
        for fold_id in range(len(self.fold_ids)):
            filepath = gen_model_filepath("{}_{}".format(name, fold_id))
            model = load_obj(filepath)
            predictions.append(model.predict(self.X_test))
        prediction_avg = np.mean(predictions, axis=0)
        prediction_df = pd.DataFrame({"card_id": self.id_test, name: prediction_avg})
        save_submission(name, prediction_df)
        return prediction_df

    def show_folds(self):
        for fold_id, (trn, val) in enumerate(self.fold_ids):
            X_trn, y_trn = get_id_from_data(self.X_train, trn), self.y_train[trn]
            X_val, y_val = get_id_from_data(self.X_train, val), self.y_train[val]
            print_info("fold", fold_id)
            print_info("X_trn.shape", get_shape(X_trn))
            print_info("y_trn.shape", y_trn.shape)
            print_info("y_trn.mean", np.mean(y_trn))
            print_info("X_val.shape", get_shape(X_val))
            print_info("y_val.shape", y_val.shape)
            print_info("y_val.mean", np.mean(y_val))

    def show_metric(self):
        print_info("metrics.mean", np.mean(self.metrics))
        print_info("metrics.std", np.std(self.metrics))

    def get_score(self, is_score=False):
        if is_score:
            return np.mean(self.metrics)
        return - np.mean(self.metrics)


def main():
    # X_train = np.ones((100, 5))
    # y_train = np.ones(100)
    # X_test = np.ones((1000, 5))
    # validator = KFoldValidator(X_train, y_train, X_test)
    # validator.validate(None)
    length = 500
    predictions = [np.ones(length) * float(i) * 10 for i in range(5)]
    print(np.mean(predictions, axis=0))


if __name__ == '__main__':
    main()
