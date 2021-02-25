from MacroF1 import macro_weighted_f1
from NeuralNetwork import NNmodel, standardize_data
from XGBoost import RandomForest
import pandas as pd
import numpy as np

# Data settings
balanced = True

# Importing train data
if not balanced:
    X_train = pd.read_csv("data/train_test_frames/final_train_x.csv")
    X_train = X_train.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
    X_train = X_train.iloc[:, 1:]
    #X_train_stand = standardize_data(X_train)
    Y_train = pd.read_csv("data/train_test_frames/final_train_y.csv")
    Y_train = Y_train.drop(columns={'Unnamed: 0'})
    Y_train = Y_train.iloc[:, 1:]
else:
    # Depends on the predicted variable
    X_train = 0
    Y_train = 0

# Importing test data
X_test = pd.read_csv("data/train_test_frames/final_test_x.csv")
X_test = X_test.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
X_test_stand = standardize_data(X_test)
Y_test = pd.read_csv("data/train_test_frames/final_test_y.csv")
Y_test = Y_test.drop(columns={'Unnamed: 0'})
dep_vars = Y_test.columns

# For loop over all dependent variables
for i in range(1, 4):
    criteria = dep_vars[i]
    depend_test = Y_test[criteria]
    print("----------------------------------------------------------------------")
    print("Dependent variable to be predicted is", criteria)
    if balanced:
        print('Data that is used is balanced')
    else:
        print('Data that is used is imbalanced')

    # Two-step binary classification for onTimeDelivery
    if criteria == 'onTimeDelivery':
        # Step 1
        # Importing train data
        if balanced:
            X_train = pd.read_csv("data/train_test_frames/balanced_train_x_Unknown.csv")
            X_train = X_train.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
            X_train = X_train.iloc[:, 1:]
            #X_train_stand = standardize_data(X_train)
            depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_Unknown.csv")
            depend_train = depend_train.drop(columns={'Unnamed: 0'})
        else:
            depend_train = Y_train[criteria]
            depend_train = depend_train.replace(0, 1)
            depend_train = depend_train.replace({'Unknown': 0})
        depend_train = depend_train.astype(np.float32)

        # Preparing test data
        depend_test = depend_test.replace({'0.0': 1})
        depend_test = depend_test.replace({'1.0': 1})
        depend_test = depend_test.replace({'Unknown': 0})
        depend_test = depend_test.astype(np.float32)

        # Predicting known or unknown
        RF1 = RandomForest(X_train, X_test, depend_train, depend_test, 'Unknown', balanced)
        print("XGB best parameters for predicting known/unknown delivery time:", RF1.best_param)
        print("XGB macro weighted F1 score for predicting known/unknown delivery time:", RF1.score)
        #NN1 = NNmodel(X_train_stand, X_test_stand, depend_train, depend_test, 'Unknown', balanced)
        #print("NN best parameters for predicting known/unknown delivery time:", NN1.best)
        #print("NN macro weighted F1 score for predicting known/unknown delivery time:", NN1.score)
        RF_pred_known = RF1.predc
        #NN_pred_known = NN1.predc

        # Step 2
        # Importing train data
        if balanced:
            X_train_onTime = pd.read_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))
            X_train_onTime = X_train_onTime.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
            X_train_onTime = X_train_onTime.iloc[:, 1:]
            #X_train_stand = standardize_data(X_train)
            depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))
            depend_train = depend_train.drop(columns={'Unnamed: 0'})
        else:
            X_train_onTime = pd.read_csv("data/train_test_frames/final_train_x_onTimeDelivery.csv")
            X_train_onTime = X_train_onTime.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
            X_train_onTime = X_train_onTime.iloc[:, 1:]
            depend_train = pd.read_csv("data/train_test_frames/final_train_y_onTimeDelivery.csv")[criteria]
        depend_train = depend_train.astype(np.float32)

        # Preparing test data
        depend_test = Y_test[criteria]
        depend_test = depend_test[RF_pred_known == 1]
        X_test_RF = X_test[RF_pred_known == 1]
        #X_test_stand_NN = X_test_stand[NN_pred_known == 1]

        # Predicting whether on time or not
        RF2 = RandomForest(X_train_onTime, X_test_RF, depend_train, depend_test, criteria, balanced)
        print("XGB best parameters for predicting onTimeDelivery when predicted known:", RF2.best_param)
        #print("XGB macro weighted F1 score for predicting onTimeDelivery when predicted known:", RF2.score)
        #NN2 = NNmodel(X_train_stand, X_test_stand_NN, depend_train, depend_test, criteria, balanced)
        #print("NN best parameters for predicting onTimeDelivery when predicted known:", NN2.best)
        #print("XGB macro weighted F1 score for predicting onTimeDelivery when predicted known:", NN2.score)
        RF_pred_onTime = RF2.predc
        #NN_pred_onTime = NN2.prediction

        # Combining the two predictions
        final_pred_RF = RF_pred_known
        final_pred_RF = final_pred_RF.astype(object)
        final_pred_RF[RF_pred_known == 0] = 'Unknown'
        #final_pred_NN = NN_pred_known
        #final_pred_NN[NN_pred_known == 0] = 'Unknown'
        k = 0
        m = 0
        for j in range(0, len(final_pred_RF)):
            if final_pred_RF[j] == 1:
                final_pred_RF[j] = RF_pred_onTime[k]
                k = k + 1
            #if final_pred_NN[j] == 1:
                #final_pred_NN[j] = NN_pred_onTime[m]
                #m = m + 1

        # Results
        depend_test = Y_test[criteria]
        classes = [0, 1, 'Unknown']
        print("XGB macro weighted F1 score for final {0} prediction: ".format(criteria),
              macro_weighted_f1(depend_test, final_pred_RF, classes))
        #print("NN macro weighted F1 score for {0}: ".format(criteria),
        #      macro_weighted_f1(depend_test, final_pred_NN, classes))

        # Save predictions
        final_pred_RF = pd.DataFrame(final_pred_RF)
        #final_pred_NN = pd.DataFrame(final_pred_NN)
        if balanced:
            final_pred_RF.to_csv("data/predictions/XGB_balanced_final_prediction_{0}.csv".format(criteria))
            #final_pred_NN.to_csv("data/predictions/NN_balanced_prediction_{0}.csv".format(criteria))
        else:
            final_pred_RF.to_csv("data/predictions/XGB_imbalanced_final_prediction_{0}.csv".format(criteria))
            #final_pred_NN.to_csv("data/predictions/NN_imbalanced_prediction_{0}.csv".format(criteria))

    else:
        # Importing train data
        if balanced:
            X_train = pd.read_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))
            X_train = X_train.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
            X_train = X_train.iloc[:, 1:]
            # X_train_stand = standardize_data(X_train)
            depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))
            depend_train = depend_train.drop(columns={'Unnamed: 0'})
        else:
            depend_train = Y_train[criteria]
        depend_train = depend_train.astype(np.float32)
        depend_test = depend_test.astype(np.float32)

        # Predicting dependent variable with XGBoost Random Forest
        RF = RandomForest(X_train, X_test, depend_train, depend_test, criteria, balanced)
        print("XGB best parameters for {0}: ".format(criteria), RF.best_param)
        print("XGB macro weighted F1 score for {0}: ".format(criteria), RF.score)

        # Predicting dependent variable with Neural Network
        #NN = NNmodel(X_train_stand, X_test_stand, depend_train, depend_test, criteria, balanced)
        #print("NN best parameters for {0}: ".format(criteria), NN.best)
        #print("NN macro weighted F1 score for {0}: ".format(criteria), NN.score)
