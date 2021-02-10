from xgboost import XGBClassifier
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope
import numpy as np


class RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        # Initiate variables
        train_x_fold_1, train_x_fold_2, train_x_fold_3, train_x_fold_4, train_x_fold_5 = [0 for _ in range(5)]
        train_y_fold_1, train_y_fold_2, train_y_fold_3, train_y_fold_4, train_y_fold_5 = [0 for _ in range(5)]
        val_x_fold_1, val_x_fold_2, val_x_fold_3, val_x_fold_4, val_x_fold_5 = [0 for _ in range(5)]
        val_y_fold_1, val_y_fold_2, val_y_fold_3, val_y_fold_4, val_y_fold_5 = [0 for _ in range(5)]

        # Data preparation
        myList = {''}
        criteria = Y_train.dtype.names[0]
        for i in range(1, 5):
            myList.add('train_x_fold_{0}_{1}'.format(i, criteria))
            myList.add('train_y_fold_{0}_{1}'.format(i, criteria))
            myList.add('val_x_fold_{0}'.format(i))
            myList.add('val_y_fold_{0}'.format(i))

        gbl = globals()
        for toImport in myList:
            i = 1
            file = "data/train_test_frames/" + toImport + ".csv"
            if file.startswith('train_x'):
                gbl['train_x_fold_{0}'.format(i)] = pd.read_csv(file).to_numpy()
            else:
                if file.startswith('train_y'):
                    gbl['train_y_fold_{0}'.format(i)] = pd.read_csv(file).to_numpy()
                    i = i+1
                else:
                    gbl[file] = pd.read_csv(file).to_numpy()

        # Hyperparameter sets
        hyperparams = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
                       'subsample': hp.uniform('subsample', 0.3, 0.9),
                       'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                       'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 5, 1))}

        def obj_func(params):
            clf = XGBClassifier(**params, use_label_encoder=False, objective="binary:logistic", eval_metric='error')
            clf.fit(train_x_fold_1, train_y_fold_1)
            pred_y_fold_1 = clf.predict(val_x_fold_1)
            auc = roc_auc_score(val_y_fold_1, pred_y_fold_1)
            clf.fit(train_x_fold_2, train_y_fold_2)
            pred_y_fold_2 = clf.predict(val_x_fold_2)
            auc = auc + roc_auc_score(val_y_fold_2, pred_y_fold_2)
            clf.fit(train_x_fold_3, train_y_fold_3)
            pred_y_fold_3 = clf.predict(val_x_fold_3)
            auc = auc + roc_auc_score(val_y_fold_3, pred_y_fold_3)
            clf.fit(train_x_fold_4, train_y_fold_4)
            pred_y_fold_4 = clf.predict(val_x_fold_4)
            auc = auc + roc_auc_score(val_y_fold_4, pred_y_fold_4)
            clf.fit(train_x_fold_5, train_y_fold_5)
            pred_y_fold_5 = clf.predict(val_x_fold_5)
            auc = auc + roc_auc_score(val_y_fold_5, pred_y_fold_5)
            return {'loss': -auc/5, 'status': STATUS_OK}

        trials = Trials()
        self.best_param = fmin(obj_func, hyperparams, max_evals=100, algo=tpe.suggest, trials=trials,
                               rstate=np.random.RandomState(1))
        best_param_values = [x for x in self.best_param.values()]

        RF_best = XGBClassifier(n_estimators=int(best_param_values[0]), learning_rate=best_param_values[1],
                                subsample=best_param_values[2], max_depth=int(best_param_values[3]),
                                colsample_bytree=best_param_values[4], min_child_weight=int(best_param_values[5]))

        RF_best.fit(X_train, Y_train)
        self.score = RF_best.score(X_test, Y_test)
