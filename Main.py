import pandas as pd
import numpy as np
from XGBoost import RandomForest

# Importing data
X_train = pd.read_csv("data/train_test_frames/final_train_x.csv")
X_train = X_train.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
X_train = X_train.iloc[:, 1:]
X_train = X_train.to_numpy()
X_test = pd.read_csv("data/train_test_frames/final_test_x.csv")
X_test = X_test.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
X_test = X_test.to_numpy()
Y_train = pd.read_csv("data/train_test_frames/final_train_y.csv")
Y_train = Y_train.drop(columns={'Unnamed: 0'})
Y_train = Y_train.iloc[:, 1:]
Y_train = Y_train.to_numpy()
Y_test = pd.read_csv("data/train_test_frames/final_test_y.csv")
Y_test = Y_test.drop(columns={'Unnamed: 0'})
dep_vars = Y_test.columns
Y_test = Y_test.to_numpy()

# For loop over all dependent variables
for i in range(0, 4):
    criteria = dep_vars[i]
    depend_train = Y_train[:, i]
    depend_test = Y_test[:, i]

    # Predicting dependent variable with XGBoost Random Forest
    if criteria == 'onTimeDelivery':
        depend_train[depend_train == 0] = 1
        depend_train[depend_train == 'Unknown'] = 0
        depend_test[depend_test == 0] = 1
        depend_test[depend_test == 'Unknown'] = 1
        RF1 = RandomForest(X_train, X_test, depend_train, depend_test, criteria)
        pred_known = RF1.prediction


        # WORK IN PROGRESS
        # RF2 = RandomForest(X_train, X_test, depend_train, depend_test, criteria)
        # print("XGB best parameters for {0}: ".format(criteria), RF2.best_param)
        # print("XGB prediction accuracy for {0}: ".format(criteria), RF2.score)
    else:
        depend_train = depend_train.astype(np.float64)
        depend_test = depend_test.astype(np.float64)
        RF = RandomForest(X_train, X_test, depend_train, depend_test, criteria)
        print("XGB best parameters for {0}: ".format(criteria), RF.best_param)
        print("XGB prediction accuracy for {0}: ".format(criteria), RF.score)

    # Predicting dependent variable with Neural Network
    # NN = NeuralNetwork(X_train, X_test, depend_train, depend_test)
    # print("NN best parameters for {0}: ", NN.best_param).format(depend_train.dtype.names[0])
    # print("NN prediction accuracy for {0}: ", NN.score).format(depend_train.dtype.names[0])

