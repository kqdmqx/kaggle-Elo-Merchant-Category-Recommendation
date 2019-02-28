import addict
from functools import partial
# from sklearn.metrics import mean_squared_error
from bayes_opt import BayesianOptimization
from validator import KFoldValidator
from models import LGBModel, XGBModel
from my_logger import print_info
from filepath_collection import gen_optimizer_filepath
from utils import load_obj, save_obj, dict_rename


def limited(val, max_th=1.0, min_th=0.0):
    if val >= max_th:
        return max_th
    elif val <= min_th:
        return min_th
    return val


def target_function_lgbmodel(validator,
                             max_bin,
                             max_depth,
                             num_leaves,
                             min_child_samples,
                             subsample,
                             subsample_freq,
                             colsample_bytree,
                             min_gain_to_split,
                             reg_lambda,
                             reg_alpha):
    lgb_params = addict.Dict()
    lgb_params.boosting_type = "gbdt"
    lgb_params.objective = "regression"
    lgb_params.metric = "rmse"
    lgb_params.learning_rate = 0.05
    lgb_params.max_bin = int(max_bin)
    lgb_params.max_depth = int(max_depth)
    lgb_params.num_leaves = int(num_leaves)
    lgb_params.min_child_samples = int(min_child_samples)
    lgb_params.subsample = limited(subsample)
    lgb_params.subsample_freq = int(subsample_freq)
    lgb_params.colsample_bytree = limited(colsample_bytree)
    lgb_params.min_gain_to_split = min_gain_to_split
    lgb_params.reg_lambda = reg_lambda
    lgb_params.reg_alpha = reg_alpha
    lgb_params.is_unbalance = True
    lgb_params.num_boost_round = 5000
    lgb_params.early_stopping_rounds = 100
    lgb_params.verbose = -1
    lgbmodel = LGBModel(lgb_params)
    validator.validate(lgbmodel, name="temp", save_model=False)
    return validator.get_score()


def target_function_xgbmodel(validator,
                             gamma,
                             max_depth,
                             min_child_weight,
                             max_delta_step,
                             max_leaves,
                             max_bin,
                             subsample,
                             colsample_bytree,
                             colsample_bylevel,
                             param_lambda,
                             alpha,
                             scale_pos_weight):
    xgb_params = addict.Dict()
    xgb_params.booster = "gbtree"
    xgb_params.objective = "binary:logistic"
    xgb_params.tree_method = "hist"
    xgb_params.grow_policy = "lossguide"
    xgb_params.eval_metric = "auc"
    xgb_params.eta = 0.2
    xgb_params.gamma = gamma
    xgb_params.max_depth = int(max_depth)
    xgb_params.min_child_weight = int(min_child_weight)
    xgb_params.max_delta_step = int(max_delta_step)
    xgb_params.max_leaves = int(max_leaves)
    xgb_params.max_bin = int(max_bin)
    xgb_params.subsample = limited(subsample)
    xgb_params.colsample_bytree = limited(colsample_bytree)
    xgb_params.colsample_bylevel = limited(colsample_bylevel)
    xgb_params["lambda"] = param_lambda
    xgb_params.alpha = alpha
    xgb_params.scale_pos_weight = scale_pos_weight
    xgb_params.nrounds = 5000
    xgb_params.early_stopping_rounds = 100
    xgb_params.silent = True
    xgb_params.verbose = 0
    xgb_params.verbose_eval = 50
    xgbmodel = XGBModel(xgb_params)
    validator.validator(xgbmodel, name="temp", save_model=False)
    return validator.get_score()


def create_optimizer(target_function_model, X_train, y_train, space, seed=123):
    space = dict_rename(space, "lambda", "param_lambda")
    validator = KFoldValidator(id_train=None,
                               X_train=X_train,
                               y_train=y_train,
                               id_test=None,
                               X_test=None)
    target_function = partial(target_function_model, validator=validator)
    bo = BayesianOptimization(target_function, space, random_state=seed)
    return bo


class Optimizer:
    def __init__(self, target_function, config):
        self.random_seed = config.random_seed
        self.space = config.space
        self.target_function = target_function

    def set_data(self, X_train, y_train):
        self.optimizer = create_optimizer(self.target_function,
                                          X_train,
                                          y_train,
                                          self.space,
                                          self.random_seed)
        return self

    def maxmize(self, init_points, n_iter):
        self.optimizer.maximize(init_points, n_iter)
        return self

    def show(self):
        print_info("optimizer.max", self.optimizer.max)
        print_info("optimizer.res", self.optimizer.res)

    def save(self, model_name):
        filepath = gen_optimizer_filepath(model_name)
        print_info("save optimizer", filepath)
        save_obj(self, filepath)
        return self


def load_optimizer(model_name):
    filepath = gen_optimizer_filepath(model_name)
    print_info("load optimizer", filepath)
    return load_obj(filepath)


def adapt_optimizer(model_type, config):
    target_function = OPTIMIZER_FUNCTIONS[model_type]
    print_info("adapt optimizer function", target_function)
    return Optimizer(target_function, config)


def OptimizerAdaptor(model_type, config):
    return adapt_optimizer(model_type, config)


OPTIMIZER_FUNCTIONS = addict.Dict()
OPTIMIZER_FUNCTIONS.lgb = target_function_lgbmodel
OPTIMIZER_FUNCTIONS.xgb = target_function_xgbmodel
