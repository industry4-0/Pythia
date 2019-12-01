import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from sklearn.utils import shuffle

import datetime as datetime
import time as time
from matplotlib import pyplot as plt
# import matplotlib as mpl
# mpl.use('TkAgg')


class DataHandlerLstm(object):
    def __init__(self,
                 data_dir,
                 test_start_date,
                 window_size):
        """
        Data parameters initialization
        :param data_dir: str indicator for directory containing csv data file
        :param test_start_date: string date for test set
        :param window_size: int window for supervised time series data creation
        """

        current_path = os.path.abspath('.')
        if current_path.endswith('src'):
            current_path = os.path.dirname(current_path)

        if current_path.endswith('serving'):
            current_path = os.path.dirname(os.path.dirname(current_path))

        self.data_dir = os.path.join(current_path, data_dir)
        self.test_start_date = test_start_date
        self.window_size = window_size

    def read_and_process(self):
        """
        Data reader
        :return: dataframe with data
        """

        data = pd.read_csv(self.data_dir,  delimiter=";")

        data.fillna(0, inplace=True)
        data = data.sort_values(by=['time'])

        K1 = data[data['sensor'] == "Down_working_roll_DE_side_reverseHorizontal"]
        K2 = data[data['sensor'] == "Upper_working_roll_DE_side_reverseHorizontal"]
        K3 = data[data['sensor'] == "Upper_working_roll_NDE_side_horizontal"]
        K4 = data[data['sensor'] == "Down_working_roll_NDE_side_horizontal"]

        DE = K2
        del DE['sensor']

        DE.columns = ['id', 'diameter', 'time', 'acc', 'ovrl_bearing', 'shock', 'vel', 'damage']

        DE['weekDay'] = ""
        for row in DE.itertuples():
            DE.at[row.Index, 'weekDay'] = pd.to_datetime(DE.at[row.Index, 'time'], format='%Y-%m-%d %H:%M:%S').strftime(
                "%a")

        DE = DE[DE.weekDay != {'Sat', 'Sun'}]  # delete weekend records

        index_list = range(0, len(DE))
        DE.index = index_list

        DE['date'] = DE['time'].astype(str).str[0:10]

        raouloIds_array = DE['id'].unique()
        raouloIds = pd.DataFrame(raouloIds_array, columns=['id'])

        for row in raouloIds.itertuples():
            raouloIds.at[row.Index, 'numOfRecords'] = int(DE[DE['id'] == raouloIds.at[row.Index, 'id']].count()[1])
            raouloIds.at[row.Index, 'numOfDamagedRecords'] = int(
                DE[(DE['id'] == raouloIds.at[row.Index, 'id']) & (DE['damage'] == 1)].count()[1])
            raouloIds.at[row.Index, 'numOfNotDamagedRecords'] = int(
                DE[(DE['id'] == raouloIds.at[row.Index, 'id']) & (DE['damage'] == 0)].count()[1])

        raouloIds = raouloIds[(raouloIds['numOfNotDamagedRecords'] != 0)]
        raouloIds = raouloIds[(raouloIds['numOfDamagedRecords'] != 0)]

        DE = self.add_cycles(raouloIds, DE)

        DE = DE.sort_values(by=['time'])

        sc = preprocessing.MinMaxScaler()

        DE[['diameter']] = sc.fit_transform(DE[['diameter']])
        DE[['acc']] = sc.fit_transform(DE[['acc']])
        DE[['vel']] = sc.fit_transform(DE[['vel']])
        DE[['ovrl_bearing']] = sc.fit_transform(DE[['ovrl_bearing']])
        DE[['RUL']] = sc.fit_transform(DE[['RUL']])

        test_df = DE.loc[DE.date >= self.test_start_date]
        train_df = DE.loc[(DE.date < self.test_start_date)]

        # prepare data for training
        features = ['diameter', 'acc', 'RUL']

        seq_length = self.window_size

        # generate X_train
        train = train_df.as_matrix(columns=features)
        train = self.series_to_supervised(train, seq_length - 1).as_matrix()
        X_train, y_train = train[:, :seq_length*len(features)], train[:, -1]
        X_train = X_train.reshape((X_train.shape[0], seq_length, len(features)))
        print('Train shapes')
        print(X_train.shape)
        print(y_train.shape)

        # generate X_test
        test = test_df.as_matrix(columns=features)
        test = self.series_to_supervised(test, seq_length - 1).as_matrix()
        X_test, y_test = test[:, :seq_length * len(features)], test[:, -1]
        X_test = X_test.reshape((X_test.shape[0], seq_length, len(features)))
        print('Test shapes')
        print(X_test.shape)
        print(y_test.shape)

        return X_train, y_train, X_test, y_test, DE, sc

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def add_cycles(self, raouloIds, DE):
        """
        Function for adding Remaining Useful Life feature
        :param raouloIds: dataframe of raoulo ids
        :param DE: dataframe with whole dataset
        :return: dataframe of whole data with the new RUL feature
        """
        dataset = pd.DataFrame()

        for row in raouloIds.itertuples():
            raoulo_df = DE[DE['id'] == row.id]
            cycles = []
            count = 0
            dmg_flag = 0
            work_flag = 0
            for i, dmg in enumerate(raoulo_df['damage']):
                if dmg:
                    work_flag = 0
                    end = i
                    count = 0
                    if not dmg_flag:
                        temp = reversed(cycles[start:end])
                        cycles[start:end] = temp
                    dmg_flag = 1
                else:
                    dmg_flag = 0
                    if not work_flag:
                        start = i
                    work_flag = 1
                    count += 1

                cycles.append(count)
            if not dmg_flag:
                cycles = list(reversed(cycles))
            raoulo_df['RUL'] = cycles
            dataset = pd.concat([dataset, raoulo_df])

        return dataset


class DataHandlerMlp(object):
    def __init__(self,
                 data_dir,
                 test_split):
        """
        Data parameters initialization
        :param data_dir: str indicator for directory containing csv data file
        :param test_split: float percentage of test set
        """

        current_path = os.path.abspath('.')
        if current_path.endswith('src'):
            current_path = os.path.dirname(current_path)

        if current_path.endswith('serving'):
            current_path = os.path.dirname(os.path.dirname(current_path))

        self.data_dir = os.path.join(current_path, data_dir)
        self.test_split = test_split

    def read_and_process(self):
        """
        Data reader
        :return: dataframe with data
        """

        data = pd.read_csv(self.data_dir,  delimiter=";")

        data.fillna(0, inplace=True)
        data = data.sort_values(by=['time'])

        K1 = data[data['sensor'] == "Down_working_roll_DE_side_reverseHorizontal"]
        K2 = data[data['sensor'] == "Upper_working_roll_DE_side_reverseHorizontal"]
        K3 = data[data['sensor'] == "Upper_working_roll_NDE_side_horizontal"]
        K4 = data[data['sensor'] == "Down_working_roll_NDE_side_horizontal"]

        DE = K2
        del DE['sensor']

        DE.columns = ['id', 'diameter', 'time', 'acc', 'ovrl_bearing', 'shock', 'vel', 'damage']

        DE['weekDay'] = ""
        for row in DE.itertuples():
            DE.at[row.Index, 'weekDay'] = pd.to_datetime(DE.at[row.Index, 'time'], format='%Y-%m-%d %H:%M:%S').strftime(
                "%a")

        DE = DE[DE.weekDay != {'Sat', 'Sun'}]  # delete weekend records

        index_list = range(0, len(DE))
        DE.index = index_list

        DE['date'] = DE['time'].astype(str).str[0:10]

        raouloIds_array = DE['id'].unique()
        raouloIds = pd.DataFrame(raouloIds_array, columns=['id'])

        for row in raouloIds.itertuples():
            raouloIds.at[row.Index, 'numOfRecords'] = int(DE[DE['id'] == raouloIds.at[row.Index, 'id']].count()[1])
            raouloIds.at[row.Index, 'numOfDamagedRecords'] = int(
                DE[(DE['id'] == raouloIds.at[row.Index, 'id']) & (DE['damage'] == 1)].count()[1])
            raouloIds.at[row.Index, 'numOfNotDamagedRecords'] = int(
                DE[(DE['id'] == raouloIds.at[row.Index, 'id']) & (DE['damage'] == 0)].count()[1])

        sc = preprocessing.MinMaxScaler()

        DE[['diameter']] = sc.fit_transform(DE[['diameter']])
        DE[['acc']] = sc.fit_transform(DE[['acc']])
        DE[['vel']] = sc.fit_transform(DE[['vel']])
        DE[['ovrl_bearing']] = sc.fit_transform(DE[['ovrl_bearing']])

        train_df, test_df = train_test_split(DE, test_size=self.test_split, shuffle=True)
        test_df = test_df.sort_values(by=['time'])

        # prepare data for training
        features = ['diameter', 'acc', 'ovrl_bearing', 'shock', 'vel']
        # features = ['diameter', 'acc', 'vel']

        X_train = train_df.as_matrix(columns=features)
        y_train = train_df.as_matrix(columns=['damage'])
        y_train = np.squeeze(y_train)
        print('Train shapes')
        print(X_train.shape)
        print(y_train.shape)

        X_test = test_df.as_matrix(columns=features)
        y_test = test_df.as_matrix(columns=['damage'])
        y_test = np.squeeze(y_test)
        print('Test shapes')
        print(X_test.shape)
        print(y_test.shape)

        return X_train, y_train, X_test, y_test, test_df['time'], sc
