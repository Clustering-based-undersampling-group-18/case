from xgboost import XGBClassifier
import DataImbalance as di
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope
import numpy as np


class RandomForest:
    def __init__(self, X, Y):
        # Data preparation
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
        # X_train1, X_train2, X_train3, X_train4, X_train5, X_val1, X_val2, X_val3, X_val4, X_val5, Y_train1, \
        # Y_train2, Y_train3, Y_train4, Y_train5, Y_val1, Y_val2, Y_val3, Y_val4, Y_val5 = di.run()

        # Hyperparameter sets
        hyperparams = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
                       'subsample': hp.uniform('subsample', 0.3, 0.9),
                       'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                       'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 5, 1))}

        def obj_func(params):
            clf = XGBClassifier(**params, objective="binary:logistic")
            kf = KFold(n_splits=5)
            auc = cross_val_score(clf, X_train, Y_train, cv=kf, scoring='roc_auc').mean()
            # clf.fit(X_train1, Y_train1)
            # auc = clf.score(X_val1, Y_val1)
            # clf.fit(X_train2, Y_train2)
            # auc = auc + clf.score(X_val2, Y_val2)
            # clf.fit(X_train3, Y_train3)
            # auc = auc + clf.score(X_val3, Y_val3)
            # clf.fit(X_train4, Y_train4)
            # auc = auc + clf.score(X_val4, Y_val4)
            # clf.fit(X_train5, Y_train5)
            # auc = auc + clf.score(X_val5, Y_val5)
            # auc = auc/5
            return {'loss': -auc, 'status': STATUS_OK}

        trials = Trials()
        self.best_param = fmin(obj_func, hyperparams, max_evals=75, algo=tpe.suggest, trials=trials,
                               rstate=np.random.RandomState(1))
        best_param_values = [x for x in self.best_param.values()]

        RF_best = XGBClassifier(n_estimators=int(best_param_values[0]), learning_rate=best_param_values[1],
                                subsample=best_param_values[2], max_depth=int(best_param_values[3]),
                                colsample_bytree=best_param_values[4], min_child_weight=int(best_param_values[5]))

        RF_best.fit(X_train, Y_train)
        self.score = RF_best.score(X_test, Y_test)
