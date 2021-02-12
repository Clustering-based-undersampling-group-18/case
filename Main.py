from sklearn.metrics import f1_score
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
    depend_test = Y_test[:, i]

    # Two-step binary classification for onTimeDelivery
    if criteria == 'onTimeDelivery':
        continue
        # Step 1
        # Importing train data
        X_train = pd.read_csv("data/train_test_frames/balanced_train_x_unknown.csv")
        X_train = X_train.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
        X_train = X_train.iloc[:, 1:]
        X_train_stand = standardize_data(X_train)
        X_train_stand = X_train_stand.to_numpy()
        X_train = X_train.to_numpy()
        depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_unknown.csv")
        depend_train = depend_train.drop(columns={'Unnamed: 0'})
        depend_train = depend_train.to_numpy()

        # Preparing test data
        depend_test[depend_test == 0] = 1
        depend_test[depend_test == 'Unknown'] = 0

        # Predicting known or unknown
        RF1 = RandomForest(X_train, X_test, depend_train, depend_test, 'unknown')
        print("XGB best parameters for predicting known/unknown delivery time:", RF1.best_param)
        NN1 = NNmodel(X_train_stand, X_test_stand, depend_train, depend_test, 'unknown')
        print("NN best parameters for predicting known/unknown delivery time:", NN1.best)
        RF_pred_known = RF1.prediction
        NN_pred_known = NN1.prediction

        # Step 2
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

        # Preparing test data
        depend_test = Y_test[:, i][RF_pred_known == 1]
        X_test_RF = X_test[RF_pred_known == 1]
        X_test_stand_NN = X_test_stand[NN_pred_known == 1]

        # Predicting whether on time or not
        RF2 = RandomForest(X_train, X_test_RF, depend_train, depend_test, criteria)
        print("XGB best parameters for predicting onTimeDelivery when predicted known:", RF2.best_param)
        NN2 = NNmodel(X_train_stand, X_test_stand_NN, depend_train, depend_test, criteria)
        print("NN best parameters for predicting onTimeDelivery when predicted known:", NN2.best)
        RF_pred_onTime = RF2.prediction
        NN_pred_onTime = NN2.prediction

        # Combining the two predictions
        final_pred_RF = RF_pred_known
        final_pred_RF[RF_pred_known == 0] = 'Unknown'
        final_pred_NN = NN_pred_known
        final_pred_NN[NN_pred_known == 0] = 'Unknown'
        k = 0
        m = 0
        for j in range(0, len(final_pred_RF)):
            if final_pred_RF[j] == 1:
                final_pred_RF[j] = RF_pred_onTime[k]
                k = k + 1
            if final_pred_NN[j] == 1:
                final_pred_NN[j] = NN_pred_onTime[m]
                m = m + 1

        # Results
        depend_test = Y_test[:, i]
        print("XGB prediction accuracy for {0}: ".format(criteria), f1_score(depend_test, final_pred_RF))
        print("NN weighted F1 score for {0}: ".format(criteria), f1_score(depend_test, final_pred_NN))

    else:
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
        depend_train = depend_train.astype(np.float32)
        depend_test = depend_test.astype(np.float32)

        # Predicting dependent variable with XGBoost Random Forest
        RF = RandomForest(X_train, X_test, depend_train, depend_test, criteria)
        print("XGB best parameters for {0}: ".format(criteria), RF.best_param)
        print("XGB weighted F1 score for {0}: ".format(criteria), RF.score)

        # Predicting dependent variable with Neural Network
        #NN = NNmodel(X_train_stand, X_test_stand, depend_train, depend_test, criteria)
        #print("NN best parameters for {0}: ".format(criteria), NN.best)
        #print("NN prediction accuracy for {0}: ".format(criteria), NN.score)
