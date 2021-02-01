import pandas as pd
import numpy as np
from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
import pydot
import random

# Importing data
random.seed(1234)
frame = pd.read_csv("data/frame.csv")

# Splitting data
X = frame[['sellerId', 'totalPrice', 'quantityOrdered', 'countryCode', 'transporterCode', 'transporterName',
           'transporterNameOther', 'fulfilmentType', 'brickName', 'chunkName', 'productGroup', 'productSubGroup',
           'productSubSubGroup', 'registrationDateSeller', 'countryOriginSeller', 'currentCountryAvailabilitySeller']]
X = pd.get_dummies(columns=['sellerId', 'countryCode', 'productGroup'])
features = list(X.columns)
X = X.to_numpy()
Y = frame[['noCancellation', 'onTimeDelivery', 'noReturn', 'noCase']]
Y = Y.to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)


# Hyperparameter sets
hyper_param = {'n_estimators': stats.randint(150, 1000),
               'learning_rate': stats.uniform(0.01, 0.6),
               'subsample': stats.uniform(0.3, 0.9),
               'max_depth': [3, 4, 5, 6, 7, 8, 9],
               'colsample_bytree': stats.uniform(0.5, 0.9),
               'min_child_weight': [1, 2, 3, 4]}

# Grid search for the hyperparameters
RF = XGBClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
RSCV = RandomizedSearchCV(RF, param_distributions=hyper_param, cv=cv, n_iter=5, scoring='roc_auc', error_score=0,
                          verbose=3, n_jobs=-1)
print(X_val)
print(Y_val)
random_search = RSCV.fit(X_val, Y_val)

# Prints the best hyperparameters
print("Best hyperparameter values:", random_search.best_params_)

# Predicts the test set using the best model from the grid search
RF = XGBClassifier(**random_search.best_params_)
RF.fit(X_train, Y_train)
score = RF.score(X_test, Y_test)

# Prints the accuracy of the prediction
print("RF prediction accuracy:", score)

# Visualizes the fifth tree
tree = RF.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')




