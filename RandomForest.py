from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope


class RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        # Splitting the data
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)

        # Hyperparameter sets
        hyperparams = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 35, 1)),
                       'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(1)),
                       'subsample': hp.uniform('subsample', 0.3, 0.9),
                       'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
                       'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
                       'min_child_weight': scope.int(hp.uniform('min_child_weight', 1, 5, 1))}

        def obj_func(params):
            clf = XGBClassifier(**params)
            auc = cross_val_score(clf, X_train, Y_train, cv=5, scoring='roc_auc').mean()
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