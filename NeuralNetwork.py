# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:57:24 2021

@author: thijs
"""

import tensorflow as tf
from sklearn.metrics import roc_auc_score
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys


# frame = pd.read_csv("data/frame.csv")
#
# X = frame[['totalPrice',
#           'quantityOrdered',
#           'countryCode',
#           'fulfilmentType',
#           'promisedDeliveryDate',
#           'productGroup',
#           'registrationDateSeller',
#           'countryOriginSeller',
#           'currentCountryAvailabilitySeller']]
#
# X = pd.get_dummies(X, column   'fulfilmentType',
#                               'productGroup',
#                             s=['countryCode',
#                              'countryOriginSeller',
#                               'currentCountryAvailabilitySeller'])
# X = X.to_numpy()
#
# Y= frame[["noReturn"]]
#
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
# X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)

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
                    files[toImport] = temp

        space = {'choice': hp.choice('num_layers',
                                     [{'layers': 'one', },
                                      {'layers': 'two',
                                       'units2': hp.uniform('units2', 64, 1024),
                                       'dropout2': hp.uniform('dropout2', 0, .5)}
                                      ]),

                 'units1': hp.uniform('units1', 64, 1024),

                 'dropout0': hp.uniform('dropout0', 0, .5),
                 'dropout1': hp.uniform('dropout1', 0, .5),
                 'batch_size': 5,
                 'nb_epochs': 100,
                 'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
                 'momentum': hp.loguniform('momentum', np.log(0.01), np.log(1))
                 }

        def create_model(space):
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
            NNmodel.fit(files.get('train_x_fold_1'), files.get('train_y_fold_1'), epochs=space['nb_epochs'],
                        batch_size=space['batch_size'], verbose=0)
            pred_y_fold_1 = NNmodel.predict_proba(files.get('val_x_fold_1'), verbose=0)
            auc = roc_auc_score(files.get('val_y_fold_1'), pred_y_fold_1)

            NNmodel.fit(files.get('train_x_fold_2'), files.get('train_y_fold_2'), epochs=space['nb_epochs'],
                        batch_size=space['batch_size'], verbose=0)
            pred_y_fold_2 = NNmodel.predict_proba(files.get('val_x_fold_2'), verbose=0)
            auc = auc + roc_auc_score(files.get('val_y_fold_2'), pred_y_fold_2)

            NNmodel.fit(files.get('train_x_fold_3'), files.get('train_y_fold_3'), epochs=space['nb_epochs'],
                        batch_size=space['batch_size'], verbose=0)
            pred_y_fold_3 = NNmodel.predict_proba(files.get('val_x_fold_3'), verbose=0)
            auc = auc + roc_auc_score(files.get('val_y_fold_3'), pred_y_fold_3)

            NNmodel.fit(files.get('train_x_fold_4'), files.get('train_y_fold_4'), epochs=space['nb_epochs'],
                        batch_size=space['batch_size'], verbose=0)
            pred_y_fold_4 = NNmodel.predict_proba(files.get('val_x_fold_4'), verbose=0)
            auc = auc + roc_auc_score(files.get('val_y_fold_4'), pred_y_fold_4)

            NNmodel.fit(files.get('train_x_fold_5'), files.get('train_y_fold_5'), epochs=space['nb_epochs'],
                        batch_size=space['batch_size'], verbose=0)
            pred_y_fold_5 = NNmodel.predict_proba(files.get('val_x_fold_5'), verbose=0)
            auc = auc + roc_auc_score(files.get('val_y_fold_5'), pred_y_fold_5)

            sys.stdout.flush()
            return {'loss': -auc/5, 'status': STATUS_OK}

        trials = Trials()
        self.best_param = fmin(create_model, space, algo=tpe.suggest, max_evals=100, trials=trials,
                         rstate=np.random.RandomState(1))
        best_param_values = [x for x in self.best_param.values()]

        



