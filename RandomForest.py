import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import pydot
import random

# Importing data
# random.seed(1234)
frame_2019 = pd.read_csv("data/data_2019.csv")  # 2110338
frame_2020 = pd.read_csv("data/data_2020.csv")  # 2645037
frame = pd.concat([frame_2019, frame_2020], ignore_index=True)  # 4755375

# Splitting data
X = frame[['totalPrice', 'quantityOrdered', 'sellerId', 'countryCode', 'productGroup']]
X = pd.get_dummies(X)
features = list(X.columns)
X = X.to_numpy()
Y = frame[['noCancellation', 'onTimeDelivery', 'noReturn', 'noCase']]
Y = Y.to_numpy()
Y[np.isnan(Y)] = 0.5
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)


# Hyperparameter sets
hyperparam = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100],
    'max_features': [2, 3, 4],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300]
}

# Grid search for the hyperparameters
RF = RandomForestClassifier()
grid_search = GridSearchCV(estimator=RF, param_grid=hyperparam, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_val, Y_val)

# Prints the best hyperparameters
print("Best hyperparameter values:", grid_search.best_params_)

# Predicts the test set using the best model from the grid search
RF = RandomForestClassifier(**grid_search.best_params_)
RF.fit(X_train, Y_train)
score = RF.score(X_test, Y_test)

# Prints the accuracy of the prediction
print("RF prediction accuracy:", score)

# Visualizing a tree
tree = RF.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')




