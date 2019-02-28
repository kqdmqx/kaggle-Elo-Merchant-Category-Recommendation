import time


def get_curdate():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def features_downcast(name):
    return "./data/features_downcast/{}".format(name)


def gen_log_filepath(name):
    return "./log/{}_{}.log".format(name, get_curdate())


def gen_model_filepath(name):
    return "./models/{}.pkl".format(name)


def gen_optimizer_filepath(name):
    return "./optimizers/{}_{}.pkl".format(name, get_curdate())


def gen_submission_filepath(name):
    return "./data/submissions/{}.csv".format(name)


def gen_oof_filepath(name):
    return "./data/oof_downcast/{}".format(name)


# def get_cat_frequent(data, col, key="card_id"):
#     dummies = pd.get_dummies(data[col])
#     dummies["card_id"] = data[key]
#     features = dummies.groupby(key).sum()
#     features.rename(columns=lambda x: "{}_{}".format(col, x), inplace=True)
#     return features


# for col in ['authorized_flag',
#             'card_id',
#             'city_id',
#             'category_1',
#             'installments',
#             'category_3',
#             'merchant_category_id',
#             'month_lag',
#             'category_2',
#             'state_id',
#             'subsector_id',
#             'purchase_month',
#             'purchase_hour']:
#     pass


# col = "city_id"
# top40set = set(all_transactions[col].value_counts().head(40).index)
# all_transactions.loc[~all_transactions[col].isin(top40set), col] = 67373
# frequent = get_cat_frequent(all_transactions, col)
# save_cat_frequnet(frequent, col)
# del frequent
# gc.collect()
