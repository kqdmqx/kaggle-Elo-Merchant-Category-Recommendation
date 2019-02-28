
# CAT_PARAMS_INIT = [
#     "loss_function",
#     "eval_metric",
#     "use_best_model",
#     "one_hot_max_size",
#     "border_count",
#     "depth",
#     "colsample_bylevel",
#     "bagging_temperature",
#     "max_ctr_complexity",
#     "ctr_leaf_count_limit",
#     "model_size_reg",
#     "l2_leaf_reg",
#     "random_strength",
#     "metric_period",
#     "od_type",
#     "od_wait",
#     "iterations",
#     "learning_rate"
# ]


# def train_catmodel(X_train, y_train, X_valid, y_valid, cat_params):
#     valid_pool = cat.Pool(data=X_valid, label=y_valid)
#     cat_params_init = dict_filter(cat_params, CAT_PARAMS_INIT)

#     estimator = cat.CatBoostClassifier(**cat_params_init)
#     estimator.fit(
#         X_train, y_train,
#         use_best_model=True,
#         eval_set=valid_pool,
#         verbose_eval=cat_params.verbose_eval,
#         early_stopping_rounds=cat_params.early_stopping_rounds
#     )
#     return estimator


# class CatModel:
#     def __init__(self, model_params):
#         self.model_params = model_params

#     def train(self, X_train, y_train, X_valid, y_valid):
#         timer = Timer()
#         timer.start("catmodel.train")
#         self.estimator = train_catmodel(X_train, y_train, X_valid, y_valid, self.model_params)
#         timer.end("catmodel.train")
#         print_info("catmodel.__dict__", self.estimator.__dict__)
#         return self

#     def predict(self, X_test):
#         timer = Timer()
#         timer.start("catmodel.predict")
#         y_pred = self.estimator.predict_proba(X_test)[:, 1]
#         timer.end("catmodel.predict")
#         print_info("y_pred.shape", y_pred.shape)
#         return y_pred

#     def save(self, model_name, train_code):
#         filepath = gen_model_filepath(model_name, train_code)
#         print_info("save catmodel", filepath)
#         save_obj(self, filepath)
#         return self
