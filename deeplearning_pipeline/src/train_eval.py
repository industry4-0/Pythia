import json
import os
import numpy as np
import pandas as pd

from train_utils import plot_metric, plot_predictions, write_to_csv_lstm, write_to_csv_mlp
from train_utils import ModelHandlerLstm, ModelHandlerMlp
from train_utils import get_train_lstm_params, get_train_mlp_params
from data_utils import DataHandlerLstm, DataHandlerMlp

from hyperopt import fmin, tpe, Trials
from keras.models import load_model
from math import sqrt

from sklearn.preprocessing import MinMaxScaler


import warnings
# Turn off warnings
warnings.filterwarnings('ignore')


def run(args):
    """
    Function to prepare data and trigger experiments
    :param args: parameters from params.json file
    """

    if args['train_lstm']['train_flag']:
        data_handler = DataHandlerLstm(args['data_lstm']['data_dir'],
                                       args['data_lstm']['test_start_date'],
                                       args['data_lstm']['window_size'])
        args_train = args['train_lstm']
        ModelHandlerClass = ModelHandlerLstm
        get_train_params = get_train_lstm_params
    else:
        data_handler = DataHandlerMlp(args['data_mlp']['data_dir'],
                                      args['data_mlp']['test_split'])
        args_train = args['train_mlp']
        ModelHandlerClass = ModelHandlerMlp
        get_train_params = get_train_mlp_params

    X_train, y_train, X_test, y_test, data, scaler = data_handler.read_and_process()

    params = get_train_params(args_train)
    model_handler = ModelHandlerClass(args['more']['fine-tuning-file'],
                                      args['callbacks'],
                                      args['more']['gpus'],
                                      X_train, y_train,
                                      X_test, y_test)

    if args_train['hyperparameter_search']:
        trials = Trials()
        max_evals = args_train['num_hp_experiments']
        best = fmin(model_handler.run_model, params,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=max_evals,
                    return_argmin=False)
        tr_results = trials.results
        # Best model
        if args['train_lstm']['train_flag']:
            results = tr_results[np.argmin([r['loss'] for r in tr_results])]
        else:
            results = tr_results[np.argmax([r['recall'] for r in tr_results])]
        print(best)
    else:
        results = model_handler.run_model(params)

    model = results['model']
    history = results['history']
    y_pred = results['predictions']

    print("Results on test set:")

    if args['train_lstm']['train_flag']:
        print("MSE: {}".format(results['loss']))
        print("RMSE: {}".format(sqrt(results['loss'])))

        y_test = scaler.inverse_transform(np.expand_dims(y_test, axis=1))
        y_pred = scaler.inverse_transform(y_pred)

        test = data[data.time >= args['data_lstm']['test_start_date']]
        time = test.time[args['data_lstm']['window_size'] - 1:]
        model_to_save = 'RUL_lstm.h5'
        plot_metric(history, 'loss', 'val_loss')
        results_func = write_to_csv_lstm
        csv_name = 'results_lstm.csv'
    else:
        print("Accuracy: {}".format(results['accuracy']))
        print("Precision: {}".format(results['precision']))
        print("Recall: {}".format(results['recall']))
        print("Confusion matrix: {}".format(results['confusion']))
        time = data
        model_to_save = 'mlp_model.h5'
        plot_metric(history, 'accuracy', 'val_accuracy')
        results_func = write_to_csv_mlp
        csv_name = 'results_mlp.csv'

    plot_predictions(y_test, y_pred, time)

    current_path = os.path.abspath('.')
    if current_path.endswith('src'):
        current_path = os.path.dirname(current_path)

    if args['more']['save_final_model']:
        if os.path.isabs(args['more']['final_model_dir']):
            model_path = args['more']['final_model_dir']
        else:
            model_path = os.path.join(current_path, args['more']['final_model_dir'])
        model.save(os.path.join(model_path, model_to_save))

    if args['more']['results_to_csv']:

        if os.path.isabs(args['more']['csv_path']):
            csv_path = args['more']['csv_path']
        else:
            csv_path = os.path.join(current_path, args['more']['csv_path'])
        csv_file = os.path.join(csv_path, csv_name)
        if args_train['hyperparameter_search']:
            results = tr_results
        results_func(csv_file, params, args_train['hyperparameter_search'], results)


def main():
    """
    Main function
    """

    current_path = os.path.abspath('.')
    if current_path.endswith('src'):
        params_json = os.path.abspath("params.json")
    else:
        params_json = os.path.abspath("src/params.json")

    with open(params_json) as json_file:
        args = json.load(json_file)

    run(args)


if __name__ == "__main__":
    main()