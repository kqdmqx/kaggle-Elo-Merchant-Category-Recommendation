import addict
import numpy as np
from utils import save_obj, load_obj
from my_logger import Timer, print_info
from sklearn.metrics import mean_squared_error
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, Conv2D, GlobalAveragePooling2D, Reshape, Multiply, Lambda, Concatenate
from keras.layers import RepeatVector, MaxPooling2D, Flatten
from keras.models import Model
from keras.models import load_model as load_model_keras
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.preprocessing import StandardScaler
import time


def get_shape(data):
    if isinstance(data, np.ndarray):
        return data.shape
    if isinstance(data, dict):
        return {key: data[key].shape for key in data}
    return data.shape


def get_curdate():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# def make_model(keras_params):
#     max_features = 440000
#     embed_size = 256

#     main_code = Input((16,), name="main_m_code16")
#     topk_code = Input((150,), name="topk_m_code150")
#     main_value = Input((16,), name="main_m_value16")
#     topk_value = Input((150,), name="topk_m_value150")

#     ebd_layer = Embedding(max_features, embed_size)
#     main_ebd = ebd_layer(main_code)
#     topk_ebd = ebd_layer(topk_code)
#     con_ebd = Concatenate(axis=1)([main_ebd, topk_ebd])
#     con_ebd = Reshape((166, embed_size, 1))(con_ebd)
#     con_ebd = BatchNormalization()(con_ebd)
#     conv = Conv2D(64, (3, 3))(con_ebd)
#     conv = Activation("relu")(conv)
#     conv = MaxPooling2D((5, 5))(conv)
#     conv = BatchNormalization()(conv)
#     conv = Conv2D(32, (3, 3))(conv)
#     conv = Activation("relu")(conv)
#     flat_code = Flatten()(conv)

#     main_dense = Dense(8, activation="sigmoid")(main_value)
#     topk_dense = Dense(64, activation="sigmoid")(topk_value)
#     con_dense = Concatenate()([main_dense, topk_dense, flat_code])
#     con_dense = Dense(64, activation="relu")(con_dense)
#     con_dense = Dense(1, activation="linear")(con_dense)

#     model = Model(inputs=[main_code, topk_code, main_value, topk_value], outputs=con_dense)
#     model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mean_squared_error'])
#     return model


def make_model(keras_params):
    max_features = 440000
    embed_size = 64

    main_code = Input((16,), name="main_m_code16")
    topk_code = Input((150,), name="topk_m_code150")
    main_value = Input((16,), name="main_m_value16")
    topk_value = Input((150,), name="topk_m_value150")

    ebd_layer = Embedding(max_features, embed_size)
    main_ebd = ebd_layer(main_code)
    topk_ebd = ebd_layer(topk_code)
    con_ebd = Concatenate(axis=1)([main_ebd, topk_ebd])
    con_ebd = Reshape((166, embed_size, 1))(con_ebd)
    con_ebd = BatchNormalization()(con_ebd)
    conv = Conv2D(16, (3, 3))(con_ebd)
    conv = Activation("relu")(conv)
    conv = MaxPooling2D((3, 3))(conv)
    conv = BatchNormalization()(conv)
    conv = Conv2D(8, (3, 3))(conv)
    conv = Activation("relu")(conv)
    flat_code = Flatten()(conv)

    main_dense = Dense(8, activation="sigmoid")(main_value)
    topk_dense = Dense(32, activation="sigmoid")(topk_value)
    con_dense = Concatenate()([main_dense, topk_dense, flat_code])
    con_dense = Dense(16, activation="sigmoid")(con_dense)
    con_dense = Dense(1, activation="linear")(con_dense)

    model = Model(inputs=[main_code, topk_code, main_value, topk_value], outputs=con_dense)
    model.compile(loss='mean_squared_error', optimizer="adam", metrics=['mean_squared_error'])
    return model


def predict_model(estimator, X):
    prediction = estimator.predict(X)
    return prediction.reshape((get_shape(X)[0],))


def train_keras(X_train, y_train, X_valid, y_valid, keras_params):
    metric_name = "val_mean_squared_error"
    model_name = keras_params.model_name
    batch_size = keras_params.batch_size  # 512
    epochs = keras_params.epochs  # 5
    cur_time = get_curdate()
    model_filepath = "./keras_temp/{}_{}.h5".format(model_name, cur_time)
    log_filepath = "./keras_temp/{}_{}.csv".format(model_name, cur_time)
    # X_train = pad_sequences(X_train, maxlen=maxlen, padding="pre", truncating="pre")
    # X_valid = pad_sequences(X_valid, maxlen=maxlen, padding="pre", truncating="pre")

    # print_info("keras_params", keras_params)
    # print_info("keras_params.num_boost_round", keras_params.num_boost_round)
    # print_info("keras_params.early_stopping_rounds", keras_params.early_stopping_rounds)
    # print_info("keras_params.verbose_eval", keras_params.verbose_eval)

    early_stopping = EarlyStopping(monitor=metric_name, mode='min', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint(model_filepath, monitor=metric_name, mode='min', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor=metric_name, mode='min', factor=0.5, patience=5, min_lr=0.000005, verbose=1)  # patience=5, factor=0.2
    model_logger = CSVLogger(log_filepath, separator=',', append=False)

    estimator = make_model(keras_params)
    # estimator.ss = StandardScaler()
    # X_train = estimator.ss.fit_transform(X_train)
    estimator.fit(X_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(X_valid, y_valid),
                  callbacks=[early_stopping, reduce_lr, model_checkpoint, model_logger],
                  verbose=200)
    estimator = load_model_keras(model_filepath)
    p_valid = predict_model(estimator, X_valid)
    estimator.valid_metric = rmse(y_valid, p_valid)
    return estimator


class KerasModel:
    def __init__(self, model_params):
        self.model_params = model_params
        # print_info("keras.model_params", self.model_params)

    def train(self, X_train, y_train, X_valid, y_valid):
        timer = Timer()
        timer.start("keras.train")
        self.estimator = train_keras(X_train, y_train, X_valid, y_valid, self.model_params)
        timer.end("keras.train")
        print_info("keras.__dict__", self.estimator.__dict__)
        return self

    def predict(self, X_test):
        timer = Timer()
        timer.start("keras.predict")
        y_pred = predict_model(self.estimator, X_test)
        timer.end("keras.predict")
        print_info("y_pred.shape", y_pred.shape)
        return y_pred

    def get_valid_score(self):
        return float(self.estimator.valid_metric)

    def save(self, filepath):
        print_info("save keras", filepath)
        save_obj(self, filepath)
        return self


def load_model(filepath):
    print_info("load model", filepath)
    obj = load_obj(filepath)
    obj.estimator.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return obj


def adapt_model(model_type, config):
    Model = MODELS[model_type]
    print_info("adapt model", Model)
    return Model(config.model_params)


def ModelAdaptor(model_type, config):
    return adapt_model(model_type, config)


MODELS = addict.Dict()
MODELS.keras = KerasModel
