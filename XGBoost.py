from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBClassifier
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope
from sklearn.metrics import roc_auc_score
from MacroF1 import macro_weighted_f1
import numpy as np
import pandas as pd


class RandomForest:
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
                        files['train_x_fold_{0}'.format(i)] = temp
                    else:
                        files['train_y_fold_{0}'.format(i)] = temp
                else:
                    if toImport.endswith('onTimeDelivery'):
                        if toImport.__contains__('x'):
                            temp = temp.drop(columns={'Unnamed: 0.1', 'level_0'})
                            temp = temp.iloc[:, 1:]
                            files['val_x_fold_{0}'.format(i)] = temp
                        else:
                            temp = temp.drop(columns={'level_0', 'Unnamed: 0.1'})
                            temp = temp.iloc[:, 1:]
                            files['val_y_fold_{0}'.format(i)] = temp[criteria]
                            i = i + 1
                    elif toImport.__contains__('y'):
                        temp = temp.iloc[:, 1:]
                        if criteria == 'Unknown':
                            temp = temp['onTimeDelivery']
                            temp = temp.replace(0, 1)
                            temp = temp.replace({'Unknown': 0})
                            temp = temp.astype('float32')
                        else:
                            temp = temp[criteria]
                        files[toImport] = temp
                        i = i + 1
                    else:
                        temp = temp.iloc[:, 1:]
                        files[toImport] = temp

        # Hyperparameter space
        hyperparams = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
                       'subsample': hp.uniform('subsample', 0.3, 0.9),
                       'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                       'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 5, 1))}

        # Objective function for Bayesian optimization with imbalanced data
        def obj_func_imb(params):
            clf = XGBClassifier(**params, use_label_encoder=False, objective="binary:logistic", eval_metric='logloss')
            kfold = KFold(n_splits=5, random_state=1234, shuffle=True)
            auc = cross_val_score(clf, X_train, Y_train, cv=kfold, scoring='roc_auc').mean()
            return {'loss': -auc, 'status': STATUS_OK}

        # Objective function for Bayesian optimization with balanced data
        def obj_func_bal(params):
            clf = XGBClassifier(**params, use_label_encoder=False, objective="binary:logistic", eval_metric='logloss')
            clf.fit(files.get('train_x_fold_1'), files.get('train_y_fold_1'))
            pred_y_fold_1 = clf.predict(files.get('val_x_fold_1'))
            auc = roc_auc_score(files.get('val_y_fold_1'), pred_y_fold_1)
            clf.fit(files.get('train_x_fold_2'), files.get('train_y_fold_2'))
            pred_y_fold_2 = clf.predict(files.get('val_x_fold_2'))
            auc += roc_auc_score(files.get('val_y_fold_2'), pred_y_fold_2)
            clf.fit(files.get('train_x_fold_3'), files.get('train_y_fold_3'))
            pred_y_fold_3 = clf.predict(files.get('val_x_fold_3'))
            auc += roc_auc_score(files.get('val_y_fold_3'), pred_y_fold_3)
            clf.fit(files.get('train_x_fold_4'), files.get('train_y_fold_4'))
            pred_y_fold_4 = clf.predict(files.get('val_x_fold_4'))
            auc += roc_auc_score(files.get('val_y_fold_4'), pred_y_fold_4)
            clf.fit(files.get('train_x_fold_5'), files.get('train_y_fold_5'))
            pred_y_fold_5 = clf.predict(files.get('val_x_fold_5'))
            auc += roc_auc_score(files.get('val_y_fold_5'), pred_y_fold_5)
            return {'loss': -auc/5, 'status': STATUS_OK}

        # Obtaining the parameterset that maximizes the evaluation metric
        trials = Trials()
        if balanced:
            self.best_param = fmin(obj_func_bal, hyperparams, max_evals=100, algo=tpe.suggest, trials=trials,
                                   rstate=np.random.RandomState(1))
        else:
            self.best_param = fmin(obj_func_imb, hyperparams, max_evals=100, algo=tpe.suggest, trials=trials,
                                   rstate=np.random.RandomState(1))
        best_param_values = [x for x in self.best_param.values()]

        # Training the model with the best parameter values
        RF_best = XGBClassifier(n_estimators=int(best_param_values[4]), learning_rate=best_param_values[1],
                                subsample=best_param_values[5], max_depth=int(best_param_values[2]),
                                colsample_bytree=best_param_values[0], min_child_weight=int(best_param_values[3]),
                                use_label_encoder=False, objective="binary:logistic", eval_metric='logloss')
        RF_best.fit(X_train, Y_train)

        # Predicting the dependent variable with the test set
        self.predc = RF_best.predict(X_test)
        self.predp = RF_best.predict_proba(X_test)
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
