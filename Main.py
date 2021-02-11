import pandas as pd
from XGBoost import RandomForest

# Importing data
X_train = pd.read_csv("data/train_test_frames/final_train_x.csv")
X_train = X_train.drop(columns={'index', 'sellerId', 'orderDate', 'Unnamed: 0'})
X_train.to_numpy()
X_test = pd.read_csv("data/train_test_frames/final_test_x.csv")
X_test = X_test.drop(columns={'index', 'sellerId', 'orderDate', 'Unnamed: 0'})
X_test.to_numpy()
Y_train = pd.read_csv("data/train_test_frames/final_train_y.csv").to_numpy()
Y_test = pd.read_csv("data/train_test_frames/final_test_y.csv").to_numpy()


# For loop over all dependent variables
for (depend_train, depend_test) in (Y_train.T, Y_test.T):
    depend_train = depend_train.T
    depend_test = depend_test.T
    print(depend_train.dtype.names[0])

    # Predicting dependent variable with XGBoost Random Forest
    RF = RandomForest(X_train, X_test, depend_train, depend_test)
    print("XGB best parameters for {0}: ", RF.best_param).format(depend_train.dtype.names[0])
    print("XGB prediction accuracy for {0}: ", RF.score).format(depend_train.dtype.names[0])

    # Predicting dependent variable with Neural Network
    # NN = NeuralNetwork(X_train, X_test, depend_train, depend_test)
    # print("NN best parameters for {0}: ", NN.best_param).format(depend_train.dtype.names[0])
    # print("NN prediction accuracy for {0}: ", NN.score).format(depend_train.dtype.names[0])

