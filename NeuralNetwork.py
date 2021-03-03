# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:57:24 2021

@author: thijs
"""

import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from MacroF1 import macro_weighted_f1
import numpy as np
import pandas as pd
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
    def __init__(self, X_train, X_test, Y_train, Y_test, criteria, balanced):

        if balanced:
            # Creating a list with all the file names that have to be imported
            names = []
            for i in range(1, 6):
                names.append('train_x_fold_{0}_{1}'.format(i, criteria))
                names.append('train_y_fold_{0}_{1}'.format(i, criteria))
                if criteria == 'onTimeDelivery':
                    names.append('val_x_fold_{0}_{1}'.format(i, criteria))
                    names.append('val_y_fold_{0}_{1}'.format(i, criteria))
                else:
                    names.append('val_x_fold_{0}'.format(i))
                    names.append('val_y_fold_{0}'.format(i))

            # Importing the files mentioned in the list
            i = 1
            files = {}
            for toImport in names:
                file = "data/train_test_frames/" + toImport + ".csv"
                temp = pd.read_csv(file)
                temp = temp.drop(columns={'Unnamed: 0'})
                if toImport.startswith('train'):
                    if toImport.startswith('train_x'):
                        temp = temp.iloc[:, 1:]
                        files['train_x_fold_{0}'.format(i)] = temp.astype(np.float32)
                    else:
                        files['train_y_fold_{0}'.format(i)] = temp.astype(np.float32)
                else:
                    if toImport.endswith('onTimeDelivery'):
                        if toImport.__contains__('x'):
                            temp = temp.drop(columns={'Unnamed: 0.1', 'level_0'})
                            temp = temp.iloc[:, 1:]
                            files['val_x_fold_{0}'.format(i)] = temp.astype(np.float32)
                        else:
                            temp = temp.drop(columns={'level_0', 'Unnamed: 0.1'})
                            temp = temp.iloc[:, 1:]
                            files['val_y_fold_{0}'.format(i)] = temp[criteria].astype(np.float32)
                            i = i + 1
                    elif toImport.__contains__('y'):
                        temp = temp.iloc[:, 1:]
                        if criteria == 'Unknown':
                            temp = temp['onTimeDelivery']
                            temp = temp.replace({'0.0': 1})
                            temp = temp.replace({'1.0': 1})
                            temp = temp.replace({'Unknown': 0})
                            temp = temp.astype('float32')
                        else:
                            temp = temp[criteria]
                        files[toImport] = temp.astype(np.float32)
                        i = i + 1
                    else:
                        temp = temp.iloc[:, 1:]
                        files[toImport] = temp.astype(np.float32)

        else:

            X_trainsub, X_val, Y_trainsub, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)

        batch1 = int((len(X_train) * 0.8) / 100)
        batch2 = int((len(X_train) * 0.8) / 50)
        batch3 = int((len(X_train) * 0.8) / 10)

        # Hyperparameter space
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

            # loss, accuracy = NNmodel.evaluate(Xv, Yv, verbose=0)
            predict = NNmodel.predict(Xv, verbose=0)
            return roc_auc_score(Yv, predict)

        # Objective function for Bayesian optimization with imbalanced data
        def obj_func_imb(space):
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
            self.history = NNmodel.fit(X_trainsub, Y_trainsub,
                                       validation_data=(X_val, Y_val),
                                       epochs=space['nb_epochs'],
                                       batch_size=space['batch_size'],
                                       verbose=0)

            loss, accuracy = NNmodel.evaluate(X_val, Y_val, verbose=0)
            predict = NNmodel.predict_proba(X_val, verbose=0)
            roc_auc = roc_auc_score(Y_val, predict)
            print('AUC:', roc_auc)
            sys.stdout.flush()
            return {'loss': -roc_auc, 'status': STATUS_OK}

        # Objective function for Bayesian optimization with balanced data
        def obj_func_bal(space):
            auc = train_model(space, files.get('train_x_fold_1'), files.get('train_y_fold_1'),
                              files.get('val_x_fold_1'), files.get('val_y_fold_1'))
            auc += train_model(space, files.get('train_x_fold_2'), files.get('train_y_fold_2'),
                               files.get('val_x_fold_2'), files.get('val_y_fold_2'))
            auc += train_model(space, files.get('train_x_fold_3'), files.get('train_y_fold_3'),
                               files.get('val_x_fold_3'), files.get('val_y_fold_3'))
            auc += train_model(space, files.get('train_x_fold_4'), files.get('train_y_fold_4'),
                               files.get('val_x_fold_4'), files.get('val_y_fold_4'))
            auc += train_model(space, files.get('train_x_fold_5'), files.get('train_y_fold_5'),
                               files.get('val_x_fold_5'), files.get('val_y_fold_5'))
            auc = auc/5
            print('AUC:', auc)
            sys.stdout.flush()
            return {'loss': -auc, 'status': STATUS_OK}

        # Obtaining the parameterset that maximizes the evaluation metric
        trials = Trials()
        if balanced:
            self.best = fmin(obj_func_bal, space, algo=tpe.suggest, max_evals=1, trials=trials,
                             rstate=np.random.RandomState(1))
        else:
            self.best = fmin(obj_func_imb, space, algo=tpe.suggest, max_evals=1, trials=trials,
                             rstate=np.random.RandomState(1))
        print('best: ', self.best)

        batch1 = int(len(X_train) / 100)
        batch2 = int(len(X_train) / 50)
        batch3 = int(len(X_train) / 10)

        # Training the model with the best parameter values
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
            self.history = NNmodel.fit(X_train, Y_train, epochs=100, batch_size=batch1, verbose=0)
        elif self.best['batch_size'] == 1:
            self.history = NNmodel.fit(X_train, Y_train,  epochs=100, batch_size=batch2, verbose=0)
        else:
            self.history = NNmodel.fit(X_train, Y_train, epochs=100, batch_size=batch3, verbose=0)

        # Predicting the dependent variable with the test set
        self.predc = NNmodel.predict_classes(X_test, verbose=0)
        self.predp = NNmodel.predict_proba(X_test, verbose=0)
        framec = pd.DataFrame(self.predc)
        framep = pd.DataFrame(self.predp)
        if balanced:
            framec.to_csv("data/predictions/XGB_balanced_c_prediction_{0}.csv".format(criteria))
            framep.to_csv("data/predictions/XGB_balanced_p_prediction_{0}.csv".format(criteria))
        else:
            framec.to_csv("data/predictions/XGB_imbalanced_c_prediction_{0}.csv".format(criteria))
            framep.to_csv("data/predictions/XGB_imbalanced_p_prediction_{0}.csv".format(criteria))
        if criteria != 'onTimeDelivery':
            self.score = macro_weighted_f1(Y_test, self.predc, [0, 1])
