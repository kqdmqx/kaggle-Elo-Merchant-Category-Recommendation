import addict
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from utils import save_obj, load_obj
from my_logger import Timer, print_info
from sklearn.metrics import mean_squared_error


def train_lgbmodel(X_train, y_train, X_valid, y_valid, lgb_params):
    lgb_data_train = lgb.Dataset(data=X_train, label=y_train)
    lgb_data_valid = lgb.Dataset(data=X_valid, label=y_valid)

    # print_info("lgb_params", lgb_params)
    # print_info("lgb_params.num_boost_round", lgb_params.num_boost_round)
    # print_info("lgb_params.early_stopping_rounds", lgb_params.early_stopping_rounds)
    # print_info("lgb_params.verbose_eval", lgb_params.verbose_eval)

    estimator = lgb.train(
        lgb_params,
        lgb_data_train,
        valid_sets=[lgb_data_train, lgb_data_valid],
        valid_names=["data_train", "data_valid"],
        num_boost_round=lgb_params.num_boost_round,
        early_stopping_rounds=lgb_params.early_stopping_rounds,
        verbose_eval=lgb_params.verbose_eval
    )
    return estimator


def get_valid_score_lgbmodel(estimator, metric_name):
    return estimator.best_score["data_valid"][metric_name]


class LGBModel:
    def __init__(self, model_params):
        self.model_params = model_params
        # print_info("lgbmodel.model_params", self.model_params)

    def train(self, X_train, y_train, X_valid, y_valid):
        timer = Timer()
        timer.start("lgbmodel.train")
        self.estimator = train_lgbmodel(X_train, y_train, X_valid, y_valid, self.model_params)
        timer.end("lgbmodel.train")
        print_info("lgbmodel.__dict__", self.estimator.__dict__)
        return self

    def predict(self, X_test):
        timer = Timer()
        timer.start("lgbmodel.predict")
        if self.model_params.objective == "regression":
            y_pred = self.estimator.predict(X_test)
        # if self.model_params.objective == "binary":
        #     y_pred = self.estimator.predict_proba(X_test)[:, 1]
        else:
            y_pred = self.estimator.predict(X_test)
        timer.end("lgbmodel.predict")
        print_info("y_pred.shape", y_pred.shape)
        return y_pred

    def get_valid_score(self):
        return get_valid_score_lgbmodel(self.estimator, metric_name=self.model_params.metric)

    def save(self, filepath):
        print_info("save lgbmodel", filepath)
        save_obj(self, filepath)
        return self


def train_xgbmodel(X_train, y_train, X_valid, y_valid, xgb_params):
    xgb_data_train = xgb.DMatrix(X_train, label=y_train)
    xgb_data_valid = xgb.DMatrix(X_valid, label=y_valid)
    estimator = xgb.train(
        params=xgb_params,
        dtrain=xgb_data_train,
        evals=[(xgb_data_valid, 'valid')],
        num_boost_round=xgb_params.nrounds,
        early_stopping_rounds=xgb_params.early_stopping_rounds,
        verbose_eval=xgb_params.verbose_eval,
        feval=None
    )
    return estimator


def predict_xgbmodel(estimator, X_test):
    xgb_data_test = xgb.DMatrix(X_test)
    return estimator.predict(xgb_data_test, ntree_limit=estimator.best_ntree_limit)


class XGBModel:
    def __init__(self, model_params):
        self.model_params = model_params

    def train(self, X_train, y_train, X_valid, y_valid):
        timer = Timer()
        timer.start("xgbmodel.train")
        self.estimator = train_xgbmodel(X_train, y_train, X_valid, y_valid, self.model_params)
        timer.end("xgbmodel.train")
        print_info("xgbmodel.__dict__", self.estimator.__dict__)
        return self

    def predict(self, X_test):
        timer = Timer()
        timer.start("xgbmodel.predict")
        y_pred = predict_xgbmodel(self.estimator, X_test)
        timer.end("xgbmodel.predict")
        print_info("y_pred.shape", y_pred.shape)
        return y_pred

    def get_valid_score(self):
        return float(self.estimator.best_score)

    def save(self, filepath):
        print_info("save xgbmodel", filepath)
        save_obj(self, filepath)
        return self


def train_lrmodel(X_train, y_train, X_valid, y_valid, model_params):
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    estimator.fit(X_train, y_train)
    # if model_params.metric == "rmse"
    estimator.train_metric = np.sqrt(mean_squared_error(estimator.predict(X_train), y_train))
    estimator.valid_metric = np.sqrt(mean_squared_error(estimator.predict(X_valid), y_valid))
    return estimator


def predict_lrmodel(estimator, X_test):
    return estimator.predict(X_test)


class LRModel:
    def __init__(self, model_params):
        self.model_params = model_params

    def train(self, X_train, y_train, X_valid, y_valid):
        timer = Timer()
        timer.start("lrmodel.train")
        self.estimator = train_lrmodel(X_train, y_train, X_valid, y_valid, self.model_params)
        timer.end("lrmodel.train")
        print_info("lrmodel.__dict__", self.estimator.__dict__)
        return self

    def predict(self, X_test):
        timer = Timer()
        timer.start("lrmodel.predict")
        y_pred = predict_lrmodel(self.estimator, X_test)
        timer.end("lrmodel.predict")
        print_info("y_pred.shape", y_pred.shape)
        return y_pred

    def get_valid_score(self):
        return float(self.estimator.valid_metric)

    def save(self, filepath):
        print_info("save lrmodel", filepath)
        save_obj(self, filepath)
        return self


def train_rfrmodel(X_train, y_train, X_valid, y_valid, model_params):
    from sklearn.ensemble import RandomForestRegressor
    estimator = RandomForestRegressor(**model_params)
    estimator.fit(X_train, y_train)
    # if model_params.metric == "rmse"
    estimator.train_metric = np.sqrt(mean_squared_error(estimator.predict(X_train), y_train))
    estimator.valid_metric = np.sqrt(mean_squared_error(estimator.predict(X_valid), y_valid))
    return estimator


def predict_rfrmodel(estimator, X_test):
    return estimator.predict(X_test)


class RFRModel:
    def __init__(self, model_params):
        self.model_params = model_params

    def train(self, X_train, y_train, X_valid, y_valid):
        timer = Timer()
        timer.start("rfrmodel.train")
        self.estimator = train_rfrmodel(X_train, y_train, X_valid, y_valid, self.model_params)
        timer.end("rfrmodel.train")
        print_info("rfrmodel.__dict__", self.estimator.__dict__)
        return self

    def predict(self, X_test):
        timer = Timer()
        timer.start("rfrmodel.predict")
        y_pred = predict_rfrmodel(self.estimator, X_test)
        timer.end("rfrmodel.predict")
        print_info("y_pred.shape", y_pred.shape)
        return y_pred

    def get_valid_score(self):
        return float(self.estimator.valid_metric)

    def save(self, filepath):
        print_info("save rfrmodel", filepath)
        save_obj(self, filepath)
        return self


def load_model(filepath):
    print_info("load model", filepath)
    return load_obj(filepath)


def adapt_model(model_type, config):
    Model = MODELS[model_type]
    print_info("adapt model", Model)
    return Model(config.model_params)


def ModelAdaptor(model_type, config):
    return adapt_model(model_type, config)


MODELS = addict.Dict()
MODELS.lgb = LGBModel
MODELS.xgb = XGBModel
MODELS.lr = LRModel
MODELS.rfr = RFRModel
