"""
This script contains the function for computing forecasts with a Neural Network model
"""

import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from MacroF1 import macro_weighted_f1_print
import numpy as np
import pandas as pd
import sys


class NNmodel:
    def __init__(self, x_train, x_test, y_train, y_test, criteria, balanced):

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
            #x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1234)

        batch1 = 128
        batch2 = 256
        batch3 = 512

        # Hyperparameter space
        hyperparams = {'choice': hp.choice('num_layers',
                                           [{'layers': 'one', },
                                            {'layers': 'two',
                                             'units2': hp.uniform('units2', 64, 1024),
                                             'dropout2': hp.uniform('dropout2', 0, .5)}
                                            ]),

                       'units1': hp.uniform('units1', 64, 1024),
                       'dropout0': hp.uniform('dropout0', 0, .5),
                       'dropout1': hp.uniform('dropout1', 0, .5),
                       'batch_size': hp.choice('batch_size', [batch1, batch2, batch3]),
                       'nb_epochs': 10,
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
                       'momentum': hp.loguniform('momentum', np.log(0.01), np.log(1))
                       }

        def train_model(space, xt, yt, xv, yv):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dropout(space['dropout0']))
            model.add(tf.keras.layers.Dense(space['units1'], activation='relu'))
            model.add(tf.keras.layers.Dropout(space['dropout1']))

            if space['choice']['layers'] == 'two':
                model.add(tf.keras.layers.Dense(space['choice']['units2'], activation='relu'))
                model.add(tf.keras.layers.Dropout(space['choice']['dropout2']))

            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            opt = tf.keras.optimizers.SGD(lr=space['learning_rate'], momentum=space['momentum'])
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
            self.history = model.fit(xt, yt, epochs=space['nb_epochs'], batch_size=space['batch_size'], verbose=0)

            predict = model.predict(xv, verbose=0)
            predict = (predict > 0.5).astype("int32")
            return roc_auc_score(yv, predict)

        """# Objective function for Bayesian optimization with imbalanced data
        def obj_func_imb(space):
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dropout(space['dropout0']))
            model.add(tf.keras.layers.Dense(space['units1'], activation='relu'))
            model.add(tf.keras.layers.Dropout(space['dropout1']))

            if space['choice']['layers'] == 'two':
                model.add(tf.keras.layers.Dense(space['choice']['units2'], activation='relu'))
                model.add(tf.keras.layers.Dropout(space['choice']['dropout2']))

            model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

            opt = tf.keras.optimizers.SGD(lr=space['learning_rate'], momentum=space['momentum'])
            model.compile(optimizer=opt, loss='binary_crossentropy', metrics=["accuracy"])
            self.history = model.fit(x_train2, y_train2, validation_data=(x_val, y_val), epochs=space['nb_epochs'],
                                     batch_size=space['batch_size'], verbose=0)

            predict = model.predict(x_val, verbose=0)
            predict = (predict > 0.5).astype("int32")
            roc_auc = roc_auc_score(y_val, predict)
            sys.stdout.flush()
            return {'loss': -roc_auc, 'status': STATUS_OK}"""

        # Objective function for Bayesian optimization with imbalanced data
        def obj_func_imb(params):
            auc = 0
            kfold = KFold(n_splits=5, random_state=1234, shuffle=True)
            for train_index, test_index in kfold.split(x_train):
                x_train_fold, x_val_fold = x_train[train_index], x_train[test_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[test_index]
                auc += train_model(params, x_train_fold, y_train_fold, x_val_fold, y_val_fold)
            return {'loss': -auc, 'status': STATUS_OK}

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
            sys.stdout.flush()
            return {'loss': -auc, 'status': STATUS_OK}

        # Obtaining the parameter set that maximizes the evaluation metric
        trials = Trials()
        if balanced:
            self.best = fmin(obj_func_bal, hyperparams, algo=tpe.suggest, max_evals=1, trials=trials,
                             rstate=np.random.RandomState(1))
        else:
            self.best = fmin(obj_func_imb, hyperparams, algo=tpe.suggest, max_evals=1, trials=trials,
                             rstate=np.random.RandomState(1))

        # Training the model with the best parameter values
        nn = tf.keras.models.Sequential()
        nn.add(tf.keras.layers.Dropout(self.best['dropout0']))
        nn.add(tf.keras.layers.Dense(self.best['units1'], activation='relu'))
        nn.add(tf.keras.layers.Dropout(self.best['dropout1']))

        if self.best['num_layers'] == 1:
            nn.add(tf.keras.layers.Dense(self.best['units2'], activation='relu'))
            nn.add(tf.keras.layers.Dropout(self.best['dropout2']))

        nn.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        sgd = tf.keras.optimizers.SGD(lr=self.best['learning_rate'], momentum=self.best['momentum'])
        nn.compile(optimizer=sgd, loss='binary_crossentropy', metrics=["accuracy"])
        if self.best['batch_size'] == 0:
            self.history = nn.fit(x_train, y_train, epochs=10, batch_size=batch1, verbose=0)
        elif self.best['batch_size'] == 1:
            self.history = nn.fit(x_train, y_train,  epochs=10, batch_size=batch2, verbose=0)
        else:
            self.history = nn.fit(x_train, y_train, epochs=10, batch_size=batch3, verbose=0)

        # Predicting the dependent variable with the test set
        self.predp = nn.predict(x_test, verbose=0)
        self.predc = (self.predp > 0.5).astype("int32")
        framep = pd.DataFrame(self.predp)
        framec = pd.DataFrame(self.predc)
        if balanced:
            framec.to_csv("data/predictions/NN_balanced_c_prediction_{0}.csv".format(criteria))
            framep.to_csv("data/predictions/NN_balanced_p_prediction_{0}.csv".format(criteria))
        else:
            framec.to_csv("data/predictions/NN_imbalanced_c_prediction_{0}.csv".format(criteria))
            framep.to_csv("data/predictions/NN_imbalanced_p_prediction_{0}.csv".format(criteria))
        if criteria != 'onTimeDelivery':
            self.score = macro_weighted_f1_print(y_test, self.predc, [0, 1])
