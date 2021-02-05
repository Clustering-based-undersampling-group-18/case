from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold


class RandomForest:
    def __init__(self, X_train, X_test, Y_train, Y_test):
        # Splitting the data
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)

        # Hyperparameter sets
        hyper_param = {'n_estimators': stats.randint(150, 1000),
                       'learning_rate': stats.uniform(0.01, 0.6),
                       'subsample': stats.uniform(0.3, 0.9),
                       'max_depth': [3, 4, 5, 6, 7, 8, 9],
                       'colsample_bytree': stats.uniform(0.5, 0.9),
                       'min_child_weight': [1, 2, 3, 4]}

        # Grid search for the hyperparameters
        RF = XGBClassifier(use_label_encoder=False)
        cv = KFold(n_splits=5, random_state=1)
        RSCV = RandomizedSearchCV(RF, param_distributions=hyper_param, cv=cv, n_iter=5, scoring='roc_auc',
                                  error_score=0, verbose=3, n_jobs=-1)
        random_search = RSCV.fit(X_val, Y_val)
        self.best_params_ = random_search.best_params_

        # Predicts the test set using the best model from the grid search
        RF = XGBClassifier(**random_search.best_params_)
        RF.fit(X_train, Y_train)
        self.score = RF.score(X_test, Y_test)
