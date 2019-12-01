from __future__ import print_function

import sys
sys.path.append('..')
import requests
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import data_utils
from train_utils import plot_predictions
import os
import argparse


# The server URL specifies the endpoint of your server running the LSTM
# model with the name "pythia" and using the predict interface.
SERVER_URL = 'http://localhost:8501/v1/models/pythia:predict'


def main(args):

    current_path = os.path.abspath('.')
    if current_path.endswith('serving_lstm'):
        current_path = os.path.dirname(os.path.dirname(current_path))

    input_file = os.path.join(current_path, args.input_file)

    temp_date = "2019-04-03"
    window_size = 100
    data_handler = data_utils.DataHandlerLstm(input_file,
                                              temp_date,
                                              window_size)

    X_1, y_1, X_2, y_2, data, scaler = data_handler.read_and_process()

    X = np.concatenate((X_1, X_2))
    y = np.concatenate((y_1, y_2))

    predict_request = json.dumps({"signature_name": "serving_default", "instances": X.tolist()})

    # Send few requests to warm-up the model.
    for _ in range(3):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()

    # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 1
    for i in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions']
        print('Avg latency: {} ms'.format((total_time * 1000) / (i + 1)))

    y_pred = np.asarray(prediction)
    y_test = scaler.inverse_transform(np.expand_dims(y, axis=1))
    y_pred = scaler.inverse_transform(y_pred)

    time = data.time[window_size - 1:]
    plot_predictions(y_test, y_pred, time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help='Path to input test data')

    args = parser.parse_args()
    main(args)