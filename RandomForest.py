import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz
from sklearn.model_selection import GridSearchCV
import pydot
import random

# Importing data
frame_2019 = pd.read_csv("data/data_2019.csv")  # 2110338
frame_2020 = pd.read_csv("data/data_2020.csv")  # 2645037
frame = pd.concat([frame_2019, frame_2020], ignore_index=True)  # 4755375
random.seed(1234)
frame = frame.sample(n=10000)

# Splitting data
X = frame[['totalPrice', 'quantityOrdered', 'countryCode']]
X = pd.get_dummies(X)
features = list(X.columns)
X = X.to_numpy()
# X[np.isnan(X)] = 0
Y = frame['noReturn']
Y = Y.to_numpy()
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1234)

# Hyperparameter sets
hyperparam = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

# Grid search for the hyperparameters
RF = RandomForestClassifier()
grid_search = GridSearchCV(estimator=RF, param_grid=hyperparam, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(train_X, train_Y)

# Prints the best hyperparameters
print("Best hyperparameter values:", grid_search.best_params_)

# Predicts the test set using the best model from the grid search
RF = grid_search.best_estimator_
score = RF.score(test_X, test_Y)

# Prints the accuracy of the prediction
print("RF prediction accuracy:", score)

# Visualizing a tree
tree = RF.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')




