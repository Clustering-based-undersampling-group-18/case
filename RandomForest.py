from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
from hyperopt.pyll import scope


class RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        # Hyperparameter sets
        hyperparams = {'n_estimators': scope.int(hp.quniform('n_estimators', 5, 100, 5)),
                       'max_features': scope.int(hp.quniform('max_features', 1, 10, 1)),
                       'max_depth': scope.int(hp.quniform('max_depth', 5, 15, 1)),
                       'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 10, 1)),
                       'min_samples_split': scope.int(hp.quniform('min_samples_split', 5, 15, 1))}

        def obj_func(params):
            clf = RandomForestClassifier(**params)
            auc = cross_val_score(clf, X_train, Y_train, cv=5, scoring='roc_auc').mean()
            return {'loss': -auc, 'status': STATUS_OK}

        trials = Trials()
        self.best_param = fmin(obj_func, hyperparams, max_evals=100, algo=tpe.suggest, trials=trials,
                               rstate=np.random.RandomState(1))
        best_param_values = [x for x in self.best_param.values()]

        RF = RandomForestClassifier(n_estimators=int(best_param_values[0]), max_features=best_param_values[1],
                                    max_depth=int(best_param_values[2]), min_samples_leaf=best_param_values[3],
                                    min_samples_split=int(best_param_values[4]))

        RF.fit(X_train, Y_train)
        self.score = RF.score(X_test, Y_test)
