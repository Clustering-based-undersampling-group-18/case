# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:57:24 2021

@author: thijs
"""

import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys


def standardize_data(X):
    """ This function standardized/ normalizes the data, required for the KMeans algorithm"""
    columns_to_standardize = ['totalPrice', 'quantityOrdered', 'promisedDeliveryDate', 'registrationDateSeller',
                              'frequencySeller']

    data_to_standardize = X[columns_to_standardize]

    scaler = StandardScaler().fit(data_to_standardize)
    standardized_data = X.copy()
    standardized_columns = scaler.transform(data_to_standardize)
    standardized_data[columns_to_standardize] = standardized_columns
    return standardized_data


class NNmodel:
    def __init__(self, X_train, X_test, Y_train, Y_test, criteria):
        # Data preparation
        names = []
        for i in range(1, 6):
            names.append('train_x_fold_{0}_{1}'.format(i, criteria))
            names.append('train_y_fold_{0}_{1}'.format(i, criteria))
            names.append('val_x_fold_{0}'.format(i))
            names.append('val_y_fold_{0}'.format(i))

        i = 1
        files = {}
        for toImport in names:
            file = "data/train_test_frames/" + toImport + ".csv"
            temp = pd.read_csv(file)
            temp = temp.drop(columns={'Unnamed: 0'})
            if toImport.startswith('train'):
                if toImport.startswith('train_x'):
                    temp = temp.iloc[:, 1:]
                    temp = standardize_data(temp)
                    files['train_x_fold_{0}'.format(i)] = temp
                else:
                    files['train_y_fold_{0}'.format(i)] = temp
                    i = i + 1
            else:
                temp = temp.iloc[:, 1:]
                if toImport.startswith('val_y'):
                    temp = temp[criteria]
                    files[toImport] = temp
                else:
                    temp = standardize_data(temp)
                    files[toImport] = temp

        batch1 = int(len(files.get('train_x_fold_1')) / 100)
        batch2 = int(len(files.get('train_x_fold_1')) / 50)
        batch3 = int(len(files.get('train_x_fold_1')) / 10)

        space = {'choice': hp.choice('num_layers',
                                     [{'layers': 'one', },
                                      {'layers': 'two',
                                       'units2': hp.uniform('units2', 64, 1024),
                                       'dropout2': hp.uniform('dropout2', 0, .5)}
                                      ]),

                 'units1': hp.uniform('units1', 64, 1024),

                 'dropout0': hp.uniform('dropout0', 0, .5),
                 'dropout1': hp.uniform('dropout1', 0, .5),
                 'batch_size': hp.choice('batch_size', [batch1, batch2, batch3]),
                 'nb_epochs': 100,
                 'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
                 'momentum': hp.loguniform('momentum', np.log(0.01), np.log(1))
                 }

        def train_model(space, Xt, Yt, Xv, Yv):
            NNmodel = tf.keras.models.Sequential()
            NNmodel.add(tf.keras.layers.Dropout(space['dropout0']))
            NNmodel.add(tf.keras.layers.Dense(space['units1'], activation='relu'))
            NNmodel.add(tf.keras.layers.Dropout(space['dropout1']))

            if space['choice']['layers'] == 'two':
                NNmodel.add(tf.keras.layers.Dense(space['choice']['units2'], activation='relu'))
                NNmodel.add(tf.keras.layers.Dropout(space['choice']['dropout2']))

            NNmodel.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            sgd = tf.keras.optimizers.SGD(lr=space['learning_rate'], momentum=space['momentum'])
            NNmodel.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["accuracy"])
            self.history = NNmodel.fit(Xt, Yt,
                                       epochs=space['nb_epochs'],
                                       batch_size=space['batch_size'],
                                       verbose=0)

            loss, accuracy = NNmodel.evaluate(Xv, Yv, verbose=0)
            predict = NNmodel.predict_proba(Xv, verbose=0)
            return roc_auc_score(Yv, predict)

        def objective_function(space):
            roc_auc1 = train_model(space, files.get('train_x_fold_1'), files.get('train_y_fold_1'),
                                   files.get('test_x_fold_1'), files.get('train_y_fold_1'))
            roc_auc2 = train_model(space, files.get('train_x_fold_2'), files.get('train_y_fold_2'),
                                   files.get('test_x_fold_2'), files.get('train_y_fold_2'))
            roc_auc3 = train_model(space, files.get('train_x_fold_3'), files.get('train_y_fold_3'),
                                   files.get('test_x_fold_3'), files.get('train_y_fold_3'))
            roc_auc4 = train_model(space, files.get('train_x_fold_4'), files.get('train_y_fold_4'),
                                   files.get('test_x_fold_4'), files.get('train_y_fold_4'))
            roc_auc5 = train_model(space, files.get('train_x_fold_5'), files.get('train_y_fold_5'),
                                   files.get('test_x_fold_5'), files.get('train_y_fold_5'))
            roc_auc = (roc_auc1 + roc_auc2 + roc_auc3 + roc_auc4 + roc_auc5) / 5
            print('AUC:', roc_auc)
            sys.stdout.flush()
            return {'loss': -roc_auc, 'status': STATUS_OK}

        trials = Trials()
        self.best = fmin(objective_function, space, algo=tpe.suggest, max_evals=100, trials=trials,
                         rstate=np.random.RandomState(1))
        print('best: ', self.best)

        batch1 = int(len(X_train) / 100)
        batch2 = int(len(X_train) / 50)
        batch3 = int(len(X_train) / 10)

        NNmodel = tf.keras.models.Sequential()
        NNmodel.add(tf.keras.layers.Dropout(self.best['dropout0']))
        NNmodel.add(tf.keras.layers.Dense(self.best['units1'], activation='relu'))
        NNmodel.add(tf.keras.layers.Dropout(self.best['dropout1']))

        if self.best['num_layers'] == 1:
            NNmodel.add(tf.keras.layers.Dense(self.best['units2'], activation='relu'))
            NNmodel.add(tf.keras.layers.Dropout(self.best['dropout2']))

        NNmodel.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        sgd = tf.keras.optimizers.SGD(lr=self.best['learning_rate'], momentum=self.best['momentum'])
        NNmodel.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["accuracy"])
        if self.best['batch_size'] == 0:
            self.history = NNmodel.fit(X_train, Y_train,
                                       validation_data=(X_test, Y_test),
                                       epochs=100,
                                       batch_size=batch1,
                                       verbose=0)
        elif self.best['batch_size'] == 1:
            self.history = NNmodel.fit(X_train, Y_train,
                                       validation_data=(X_test, Y_test),
                                       epochs=100,
                                       batch_size=batch2,
                                       verbose=0)
        else:
            self.history = NNmodel.fit(X_train, Y_train,
                                       validation_data=(X_test, Y_test),
                                       epochs=100,
                                       batch_size=batch3,
                                       verbose=0)

        loss, accuracy = NNmodel.evaluate(X_test, Y_test, verbose=0)
        self.prediction = NNmodel.predict_classes(X_test, verbose=0)
        frame = pd.DataFrame(self.prediction)
        file_name = "data/predictions/NN_prediction_{0}.csv".format(criteria)
        frame.to_csv(file_name)
        self.score = f1_score(Y_test, self.prediction)
