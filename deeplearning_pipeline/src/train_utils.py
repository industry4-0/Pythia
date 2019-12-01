import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.dates as mdates

import keras
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model
from keras.models import load_model

from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import r2_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, log_loss, mean_squared_error

import os
import csv
import math
import pandas as pd

from hyperopt import hp, STATUS_OK


# Class for learning rate and optimizer view during training
class LRlogs(keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        """
        Prints the learning rate and the optimizer during training
        :param epoch: int num epoch
        :param logs: str for stdouts
        """
        lr = K.eval(self.model.optimizer.lr)
        print('learning_rate: {}'.format(lr))
        print('optimizer: {}'.format(self.model.optimizer))


class ModelHandler(object):
    def __init__(self,
                 fine_tuning_file,
                 callbacks,
                 gpus,
                 X_train, y_train,
                 X_test, y_test):
        """
        Model initialization function
        :param fine_tuning_file: str for hdf5 checkpoint file to fine-tune
        :param callbacks: dict with callbacks from params.json file
        :param gpus: int number of gpus to use (if we train on gpu)
        :param X_train: numpy array of features for training-validation
        :param y_train: numpy array containing the values to predict for training-validation
        :param X_test: numpy array of features for testing
        :param y_test: numpy array containing the values to predict for testing
        """

        self.fine_tuning_file = fine_tuning_file

        self.adaptive_lr = callbacks['adaptive_learning_rate']
        self.adaptive_lr_patience_epochs = callbacks['adaptive_lr_patience_epochs']
        self.adaptive_lr_decay = callbacks['adaptive_lr_decay']
        self.min_adaptive_lr = callbacks['min_adaptive_lr']

        self.early_stopping = callbacks['early_stopping']
        self.early_stopping_min_change = callbacks['early_stopping_min_change']
        self.early_stopping_patience_epochs = callbacks['early_stopping_patience_epochs']

        self.exponential_lr = callbacks['exponential_lr']
        self.num_epochs_per_decay = callbacks['num_epochs_per_decay']
        self.lr_decay_factor = callbacks['lr_decay_factor']

        self.save_model = callbacks['save_model_per_epoch']
        self.model_dir = callbacks['model_dir']
        self.epochs_per_save = callbacks['epochs_per_save']

        self.gpus = gpus

        # data
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_callbacks(self):
        """
        Callbacks initialization
        :return: list with activated callbacks
        """
        callbacks = []

        if self.adaptive_lr:
            print('***** Adaptive Learning Rate activated *****')
            patience = self.adaptive_lr_patience_epochs
            factor = self.adaptive_lr_decay
            min_lr = self.min_adaptive_lr
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                          patience=patience, min_lr=min_lr)
            callbacks.append(reduce_lr)

        if self.early_stopping:
            print('***** Early Stopping activated *****')
            min_delta = self.early_stopping_min_change
            patience = self.early_stopping_patience_epochs
            early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta,
                                                           patience=patience, verbose=0,
                                                           mode='min', baseline=None,
                                                           restore_best_weights=False)
            callbacks.append(early_stopping)

        if self.save_model:
            period = self.epochs_per_save
            filepath = os.path.join(self.model_dir, "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5")
            ckpt = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0,
                                                   save_best_only=False,
                                                   save_weights_only=False,
                                                   mode='auto', period=period)
            callbacks.append(ckpt)

        if self.exponential_lr:
            print('***** Exponential Learning Rate decay activated *****')

            def schedule(epoch, lr):
                epochs_per_decay = self.num_epochs_per_decay
                decay_factor = self.lr_decay_factor
                if epoch % epochs_per_decay == 0 and epoch != 0:
                    return lr * decay_factor
                else:
                    return lr
            exp_lr = keras.callbacks.LearningRateScheduler(schedule)
            callbacks.append(exp_lr)

        callbacks.append(LRlogs())

        return callbacks

    def get_optimizer(self, name):
        """
        Optimizer function getter
        :param name: str for optimizer's name
        :return: optimizer Keras function
        """

        if name == "adam":
            return Adam
        elif name == "sgd":
            return SGD
        elif name == "adagrad":
            return Adagrad
        elif name == "adadelta":
            return Adadelta
        elif name == "rmsprop":
            return RMSprop
        else:
            raise ValueError("You should feed for optimizer a value between: adam, sgd, adagrad, adadelta, rmsprop")


class ModelHandlerLstm(ModelHandler):
    def __init__(self,
                 fine_tuning_file,
                 callbacks,
                 gpus,
                 X_train, y_train,
                 X_test, y_test):
        """
        Model initialization function
        :param fine_tuning_file: str for hdf5 checkpoint file to fine-tune
        :param callbacks: dict with callbacks from params.json file
        :param gpus: int number of gpus to use (if we train on gpu)
        :param X_train: numpy array of features for training-validation
        :param y_train: numpy array containing the values to predict for training-validation
        :param X_test: numpy array of features for testing
        :param y_test: numpy array containing the values to predict for testing
        """

        super().__init__(fine_tuning_file,
                         callbacks,
                         gpus,
                         X_train, y_train,
                         X_test, y_test)

    def model_construction(self, activation, lstm_units, dropouts):
        """
        Graph construction
        :param activation: str indicator for activation function
        :param lstm_units: list containing lstm units
        :param dropouts: list containing dropout after each lstm unit
        :return: model Keras object
        """

        nb_features = self.X_train.shape[2]
        timestamp = self.X_train.shape[1]

        return_sequences = False if len(lstm_units) == 1 else True
        model = Sequential()
        model.add(LSTM(
            input_shape=(timestamp, nb_features),
            units=lstm_units[0],
            activation=activation,
            return_sequences=return_sequences))
        if dropouts[0]:
            model.add(Dropout(dropouts[0]))

        if len(lstm_units) > 1:
            for i, (num_units, dropout) in enumerate(zip(lstm_units[1:], dropouts[1:])):
                return_sequences = False if i == len(lstm_units[1:]) - 1 else True
                model.add(LSTM(
                    units=num_units,
                    activation=activation,
                    return_sequences=return_sequences))
                if dropout:
                    model.add(Dropout(dropout))

        model.add(Dense(units=1, activation=activation))

        return model

    def train_eval(self, params, model):
        """
        Training and evaluation process
        :param params: dict containing the training parameters
        :param model: Keras model object
        :return: Keras model and history objects, float mean squared error and
        numpy array of predictions
        """

        callbacks = self.get_callbacks()

        history = model.fit(self.X_train, self.y_train,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            validation_data=(self.X_test, self.y_test),
                            shuffle=False,
                            verbose=1,
                            callbacks=callbacks)

        # make predictions
        y_pred = model.predict(self.X_test, verbose=1, batch_size=200)

        mse = mean_squared_error(self.y_test, y_pred)

        return model, history, mse, y_pred

    def get_model(self, params):
        """
        Model getter before training
        :param params: dict of training parameters
        :return: Keras model object
        """
        if self.fine_tuning_file:
            print('Fine-tuning from {}'.format(self.fine_tuning_file))
            model = load_model(self.fine_tuning_file)
        else:
            model = self.model_construction(params['activation'], params['lstm_units'], params['dropouts'])

        if self.gpus:
            model = multi_gpu_model(model, gpus=self.gpus)

        optimizer = self.get_optimizer(params['optimizer'])

        model.compile(optimizer=optimizer(params['learning_rate']),
                      loss='mean_squared_error')

        return model

    def run_model(self, params):
        """
        Training-validation and testing process with KFold
        :param params: dict containing training parameters
        :return: dict with final results of best model
        """

        model = self.get_model(params)
        model.summary()
        model, history, mse, y_pred = self.train_eval(params, model)

        if math.isnan(mse):
            mse = 500

        return {'model': model,
                'history': history,
                'params': params,
                'loss': mse,
                'predictions': y_pred,
                'status': STATUS_OK}


class ModelHandlerMlp(ModelHandler):
    def __init__(self,
                 fine_tuning_file,
                 callbacks,
                 gpus,
                 X_train, y_train,
                 X_test, y_test):
        """
        Model initialization function
        :param fine_tuning_file: str for hdf5 checkpoint file to fine-tune
        :param callbacks: dict with callbacks from params.json file
        :param gpus: int number of gpus to use (if we train on gpu)
        :param X_train: numpy array of features for training-validation
        :param y_train: numpy array containing the values to predict for training-validation
        :param X_test: numpy array of features for testing
        :param y_test: numpy array containing the values to predict for testing
        """

        super().__init__(fine_tuning_file,
                         callbacks,
                         gpus,
                         X_train, y_train,
                         X_test, y_test)

    def model_construction(self, activation, dense_units, dropouts):
        """
        Graph construction
        :param activation: str indicator for activation function
        :return: model Keras object
        """

        model = Sequential()

        input_shape = [self.X_train.shape[-1]]
        model.add(Dense(input_shape=input_shape, units=dense_units[0], activation=activation))
        if dropouts[0]:
            model.add(Dropout(dropouts[0]))

        for units, dropout in zip(dense_units[1:], dropouts[1:]):
            model.add(Dense(units=units, activation=activation))
            if dropout:
                model.add(Dropout(dropout))

        model.add(Dense(units=1, activation='sigmoid'))
        # plot_model(model, to_file='../model/mlp_plot.png', show_shapes=True,
        #            show_layer_names=True, expand_nested=True)

        return model

    def train_eval(self, params, model):
        """
        Training and evaluation process
        :param params: dict containing the training parameters
        :param model: Keras model object
        :return: Keras model and history objects, numpy array confusion matrix,
        float accuracy, float precision, float recall, float cross entropy loss
        """

        callbacks = self.get_callbacks()

        history = model.fit(self.X_train, self.y_train,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            validation_data=(self.X_test, self.y_test),
                            shuffle=True,
                            verbose=1,
                            callbacks=callbacks)

        # make predictions and compute confusion matrix
        y_pred = model.predict_classes(self.X_test, verbose=1, batch_size=200)

        # compute precision and recall
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)

        y_probs = model.predict_proba(self.X_test, verbose=1, batch_size=200)
        loss = log_loss(self.y_test, y_probs)

        cm = pd.DataFrame(confusion_matrix(self.y_test, y_pred))

        return model, history, loss, accuracy, precision, recall, cm, y_pred

    def get_model(self, params):
        """
        Model getter before training
        :param params: dict of training parameters
        :return: Keras model object
        """
        if self.fine_tuning_file:
            print('Fine-tuning from {}'.format(self.fine_tuning_file))
            model = load_model(self.fine_tuning_file)
        else:
            model = self.model_construction(params['activation'], params['dense_units'], params['dropouts'])

        if self.gpus:
            model = multi_gpu_model(model, gpus=self.gpus)

        optimizer = self.get_optimizer(params['optimizer'])

        model.compile(optimizer=optimizer(params['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def run_model(self, params):
        """
        Training-validation and testing process with KFold
        :param params: dict containing training parameters
        :return: dict with final results of best model
        """

        model = self.get_model(params)
        model.summary()
        model, history, loss, accuracy, precision, recall, cm, y_pred = self.train_eval(params, model)

        if math.isnan(loss):
            loss = 500
        return {'model': model,
                'history': history,
                'params': params,
                'loss': loss,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'confusion': cm,
                'predictions': y_pred,
                'status': STATUS_OK}


def get_train_lstm_params(args):
    """
    Dict construction with training parameters
    :param args: dict of train field in params.json file
    :return: dict with training parameters
    """

    batch_size = args['batch_size']
    if isinstance(batch_size, list):
        batch_size = hp.choice('batch_size', batch_size)

    epochs = args['epochs']
    if isinstance(epochs, list):
        epochs = hp.choice('epochs', epochs)

    lr = args['learning_rate']
    if isinstance(lr, list):
        lr = hp.choice('learning_rate', lr)

    optimizer = args['optimizer']
    if isinstance(optimizer, list):
        optimizer = hp.choice('optimizer', optimizer)

    activation = args['activation']
    if isinstance(activation, list):
        if not any(act in ['relu', 'sigmoid', 'tanh'] for act in activation):
            raise ValueError('You should feed for activation: relu, sigmoid or tanh')
        activation = hp.choice('activation', activation)

    lstm_units = args['lstm_units']
    if isinstance(lstm_units[0], list):
        lstm_units = hp.choice('lstm_units', lstm_units)

    dropouts = args['dropouts']
    if isinstance(dropouts[0], list):
        dropouts = hp.choice('dropouts', dropouts)

    params = {'batch_size': batch_size,
              'epochs': epochs,
              'learning_rate': lr,
              'optimizer': optimizer,
              'activation': activation,
              'lstm_units': lstm_units,
              'dropouts': dropouts}

    return params


def get_train_mlp_params(args):
    """
    Dict construction with training parameters
    :param args: dict of train field in params.json file
    :return: dict with training parameters
    """

    batch_size = args['batch_size']
    if isinstance(batch_size, list):
        batch_size = hp.choice('batch_size', batch_size)

    epochs = args['epochs']
    if isinstance(epochs, list):
        epochs = hp.choice('epochs', epochs)

    lr = args['learning_rate']
    if isinstance(lr, list):
        lr = hp.choice('learning_rate', lr)

    optimizer = args['optimizer']
    if isinstance(optimizer, list):
        optimizer = hp.choice('optimizer', optimizer)

    activation = args['activation']
    if isinstance(activation, list):
        if not any(act in ['relu', 'sigmoid', 'tanh'] for act in activation):
            raise ValueError('You should feed for activation: relu, sigmoid or tanh')
        activation = hp.choice('activation', activation)

    dense_units = args['dense_units']
    if isinstance(dense_units[0], list):
        dense_units = hp.choice('dense_units', dense_units)

    dropouts = args['dropouts']
    if isinstance(dropouts[0], list):
        dropouts = hp.choice('dropouts', dropouts)

    params = {'batch_size': batch_size,
              'epochs': epochs,
              'learning_rate': lr,
              'optimizer': optimizer,
              'activation': activation,
              'dense_units': dense_units,
              'dropouts': dropouts}

    return params


def write_to_csv_lstm(csv_file, params, hps, results):
    """
    Writing final results into csv file
    :param csv_file: str indicator for exported csv file
    :param params: dict containing training parameters
    :param hps: boolean indicator for hyperparameter search
    :param results: dict containing results of best model
    """
    with open(csv_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        keys = list(params.keys())
        keys.append('mse')
        writer.writerow(keys)

        def write(results):
            values = [results['params'][key] for key in keys[:-1]]
            values.append(results['loss'])
            writer.writerow(values)
        if hps:
            for result in results:
                write(result)
        else:
            write(results)


def write_to_csv_mlp(csv_file, params, hps, results):
    """
    Writing final results into csv file
    :param csv_file: str indicator for exported csv file
    :param params: dict containing training parameters
    :param hps: boolean indicator for hyperparameter search
    :param results: dict containing results of best model
    """
    with open(csv_file, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        keys = list(params.keys())
        keys.append('accuracy')
        keys.append('precision')
        keys.append('recall')
        writer.writerow(keys)

        def write(results):
            values = [results['params'][key] for key in keys[:-3]]
            values.append(results['accuracy'])
            values.append(results['precision'])
            values.append(results['recall'])
            writer.writerow(values)
        if hps:
            for result in results:
                write(result)
        else:
            write(results)


def plot_metric(History, metric_train, metric_validation):
    """
    Train-validation metric plot
    :param History: Keras history object
    :param metric_train: str identifier for loss or accuracy on train set
    :param metric_validation: str identifier for loss or accuracy on test set
    """
    plt.plot(History.history[metric_train], 'g')
    plt.plot(History.history[metric_validation], 'b')
    plt.title('model ' + metric_train)
    plt.ylabel(metric_train)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.grid(True)
    plt.show()
    plt.close()


def plot_predictions(y_true, y_pred, time):
    """
    Predictions vs Ground-truth plot
    :param y_true: numpy array of ground truth
    :param y_pred: numpy array of predictions
    :param time: series of time
    """

    preds = pd.DataFrame(y_true, columns=['ground-truth'])
    preds['predictions'] = y_pred
    time = time.to_frame()
    time.reset_index(drop=True, inplace=True)
    preds = pd.concat([preds, time], axis=1)

    preds.plot(x='time', y=['ground-truth', 'predictions'], rot=90)
    # ax = preds.plot(x='time', y='ground-truth', style='-', rot=90)
    # preds.plot(x='time', y='predictions', ax=ax, style='.', rot=90)
    plt.title('Ground-truth vs predictions')
    plt.ylabel('RUL')
    plt.xlabel('time')
    plt.legend(['ground-truth', 'prediction'], loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.close()
