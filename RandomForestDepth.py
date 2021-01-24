import pandas as pd
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
X = frame[['totalPrice', 'quantityOrdered', 'countryCode']]
X = pd.get_dummies(X)
features = list(X.columns)
X = X.to_numpy()
# X[np.isnan(X)] = 0
Y = frame['noReturn']
Y = Y.to_numpy()
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.3, random_state=1234)

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


