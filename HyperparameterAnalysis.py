import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz
import pydot

# Importing data
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


n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []
for estimator in n_estimators:
   rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
   rf.fit(x_train, y_train)
   train_pred = rf.predict(x_train)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   train_results.append(roc_auc)
   y_pred = rf.predict(x_test)
   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
   roc_auc = auc(false_positive_rate, true_positive_rate)
   test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, ‘b’, label=”Train AUC”)
line2, = plt.plot(n_estimators, test_results, ‘r’, label=”Test AUC”)
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel(‘AUC score’)
plt.xlabel(‘n_estimators’)
plt.show()



# Initializes variables and cross-validation
kf = KFold()
n_cv = kf.get_n_splits()
bestScore = 0
bestDepth = -1

# Grid search for the max depth of the trees
for i in [5, 10, 15]:
    score = 0

    # 5-fold cross validation, split in train and validation set to find a decent max_depth and stores the average score
    for train_index, test_index in kf.split(train_X, train_Y):
        X_train, val_X = X[train_index], X[test_index]
        Y_train, val_Y = Y[train_index], Y[test_index]
        RF = RandomForestClassifier(random_state=1, max_depth=i)
        forest = RF.fit(X_train, Y_train)
        score = score + RF.score(val_X, val_Y)

    # Stores the best score yet and its corresponding parameter
    if score > bestScore:
        bestScore = score
        bestDepth = i

    # Prints the average score for max depth i
    print("Max depth:", i, ", RF average prediction accuracy:", score/n_cv)

# Prints the best average score and max depth
print("Best average score and depth RF:", bestScore/n_cv, bestDepth)

# Fits the methods on the whole training set and predicts the test set, using the best max depth from the K-fold CV
RF = RandomForestClassifier(random_state=1, max_depth=bestDepth)
RF.fit(train_X, train_Y)
score = RF.score(test_X, test_Y)

# Prints the accuracy of the prediction
print("RF prediction accuracy:", score)

# Visualizing a tree
tree = RF.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')


