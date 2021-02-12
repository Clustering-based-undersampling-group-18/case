from NeuralNetwork import NNmodel, standardize_data
from XGBoost import RandomForest
import pandas as pd
import numpy as np

# Importing test data
X_test = pd.read_csv("data/train_test_frames/final_test_x.csv")
X_test = X_test.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
X_test_stand = standardize_data(X_test)
X_test_stand = X_test_stand.to_numpy()
X_test = X_test.to_numpy()
Y_test = pd.read_csv("data/train_test_frames/final_test_y.csv")
Y_test = Y_test.drop(columns={'Unnamed: 0'})
dep_vars = Y_test.columns
Y_test = Y_test.to_numpy()

# For loop over all dependent variables
for i in range(0, 4):
    criteria = dep_vars[i]
    print("Dependent variable to be predicted is", criteria)

    # Importing train data
    X_train = pd.read_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))
    X_train = X_train.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
    X_train = X_train.iloc[:, 1:]
    X_train_stand = standardize_data(X_train)
    X_train_stand = X_train_stand.to_numpy()
    X_train = X_train.to_numpy()
    depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))
    depend_train = depend_train.drop(columns={'Unnamed: 0'})
    depend_train = depend_train.to_numpy()
    depend_test = Y_test[:, i]

    # Two-step binary classification for onTimeDelivery
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
        depend_train = depend_train.astype(np.float32)
        depend_test = depend_test.astype(np.float32)

        # Predicting dependent variable with XGBoost Random Forest
        RF = RandomForest(X_train, X_test, depend_train, depend_test, criteria)
        print("XGB best parameters for {0}: ".format(criteria), RF.best_param)
        print("XGB prediction accuracy for {0}: ".format(criteria), RF.score)

        # Predicting dependent variable with Neural Network
        NN = NNmodel(X_train_stand, X_test_stand, depend_train, depend_test, criteria)
        print("NN best parameters for {0}: ", NN.best).format(depend_train.dtype.names[0])
        print("NN prediction accuracy for {0}: ", NN.score).format(depend_train.dtype.names[0])
