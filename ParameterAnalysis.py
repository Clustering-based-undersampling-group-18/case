import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.pyplot as plt

# Importing data
frame_2019 = pd.read_csv("data/data_2019.csv")  # 2110338
frame_2020 = pd.read_csv("data/data_2020.csv")  # 2645037
frame = pd.concat([frame_2019, frame_2020], ignore_index=True)  # 4755375

# Splitting data
X = frame[['totalPrice', 'quantityOrdered', 'sellerId', 'countryCode', 'productGroup']]
X = pd.get_dummies(X)
features = list(X.columns)
X = X.to_numpy()
Y = frame[['noCancellation', 'noReturn', 'noCase']]
Y = Y.to_numpy()
# Y[np.isnan(Y)] = 2
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)

# Grid search
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100]
max_depths = np.linspace(1, 32, 32, endpoint=True)
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
max_features = list(range(1, frame.shape[1]))
train_results = []
test_results = []
for estimator in max_depths:
    RF = RandomForestClassifier(max_depth=estimator, n_jobs=-1)
    RF.fit(X_train, Y_train)
    acc = RF.score(X_train, Y_train)
    print('Accuracy train:', acc)
    train_results.append(acc)
    acc = RF.score(X_test, Y_test)
    test_results.append(acc)
    print('Accuracy test:', acc)

# Visualizing accuracy
line1, = plt.plot(n_estimators, train_results, 'b', label="Train Acc")
line2, = plt.plot(n_estimators, test_results, 'r', label="Test Acc")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy')
plt.xlabel('Maximum depth')
plt.show()
