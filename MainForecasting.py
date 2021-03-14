"""
This script can be run to generate forecasts for the four criteria and perform the match classification
Using functions from NeuralNetwork.py, XGBoost.py and PredictionMetrics.py
"""
# Packages and modules
from PredictionMetrics import macro_weighted_f1_print, threshold_search, match_classification
from ExtremeGradientBoosting import XGBmodel
from DataImbalance import standardize_data
from NeuralNetwork import NNmodel
import pandas as pd
import numpy as np

# Algorithm settings
NeuralNetwork = False
XGBoost = True
balanced_data = True
threshold = True

# Importing train data
if not balanced_data:
    X_train = pd.read_csv("data/train_test_frames/final_train_x.csv")
    X_train = X_train.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
    X_train = X_train.iloc[:, 1:]
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
X_test_stand = standardize_data(X_test).astype(np.float32)
Y_test = pd.read_csv("data/train_test_frames/final_test_y.csv")
Y_test = Y_test.drop(columns={'Unnamed: 0'})
dep_vars = Y_test.columns

# For loop over all dependent variables
for i in range(1, 2):
    criteria = dep_vars[i]
    depend_test = Y_test[criteria]
    print("----------------------------------------------------------------------")
    print("Dependent variable to be predicted is", criteria)
    if balanced_data:
        print('Data that is used is balanced')
        balanced = "balanced"
    else:
        print('Data that is used is imbalanced')
        balanced = "imbalanced"

    # Two-step binary classification for onTimeDelivery
    if criteria == 'onTimeDelivery':
        # Step 1
        # Importing train data
        if balanced_data:
            X_train = pd.read_csv("data/train_test_frames/balanced_train_x_Unknown.csv")
            X_train = X_train.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
            X_train = X_train.iloc[:, 1:]
            depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_Unknown.csv")
            depend_train = depend_train.drop(columns={'Unnamed: 0'})
        else:
            depend_train = Y_train[criteria]
            depend_train = depend_train.replace({'0.0': 1})
            depend_train = depend_train.replace({'1.0': 1})
            depend_train = depend_train.replace({'Unknown': 0})
        depend_train = depend_train.astype(np.float32)

        # Preparing test data
        depend_test = depend_test.replace({'0.0': 1})
        depend_test = depend_test.replace({'1.0': 1})
        depend_test = depend_test.replace({'Unknown': 0})
        depend_test = depend_test.astype(np.float32)

        # Predicting known or unknown
        if XGBoost:
            XGB1 = XGBmodel(X_train, X_test, depend_train, depend_test, 'Unknown', balanced)
            print("XGB best parameters for predicting known/unknown delivery time:", XGB1.best_param)
            print("XGB macro weighted F1 score for predicting known/unknown delivery time:", XGB1.score)

            # Determining the best threshold
            if threshold:
                XGB_prob_known = XGB1.predp[:, 1]
                best_threshold = threshold_search(depend_test, XGB_prob_known, "XGB Unknown", balanced)
                XGB_pred_known = np.ones(len(XGB_prob_known))
                XGB_pred_known[XGB_prob_known <= best_threshold] = 0
                print("Results after threshold optimization:")
                print("XGB macro weighted F1 score for {0} with optimized threshold:".format("Unknown"),
                      macro_weighted_f1_print(depend_test, XGB_pred_known, [0, 1]))

                # Saving new prediction
                save = pd.DataFrame(XGB_pred_known)
                save.to_csv("data/predictions/XGB_{0}_ct_prediction_Unknown.csv".format(balanced))

            else:
                XGB_pred_known = XGB1.predc

        if NeuralNetwork:
            X_train_stand = standardize_data(X_train).astype(np.float32)
            NN1 = NNmodel(X_train_stand, X_test_stand, depend_train, depend_test, 'Unknown', balanced)
            print("NN best parameters for predicting known/unknown delivery time:", NN1.best)
            print("NN macro weighted F1 score for predicting known/unknown delivery time:", NN1.score)

            # Determining the best threshold
            if threshold:
                NN_prob_known = NN1.predp
                NN_prob_known = NN_prob_known.T
                NN_prob_known = NN_prob_known[0]
                best_threshold = threshold_search(depend_test, NN_prob_known, "NN Unknown", balanced)
                NN_pred_known = np.ones(len(NN_prob_known))
                NN_pred_known[NN_prob_known <= best_threshold] = 0
                print("***RESULTS WITH THRESHOLD***")
                print("XGB macro weighted F1 score for {0} with optimized threshold: ".format("Unknown"),
                      macro_weighted_f1_print(depend_test, NN_pred_known, [0, 1]))

                # Saving new prediction
                save = pd.DataFrame(NN_pred_known)
                save.to_csv("data/predictions/NN_{0}_ct_prediction_Unknown.csv".format(balanced))

            else:
                NN_pred_known = NN1.predc
                NN_pred_known = NN_pred_known.T
                NN_pred_known = NN_pred_known[0]

        # Step 2
        # Importing train data
        if balanced_data:
            X_train_onTime = pd.read_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))
            X_train_onTime = X_train_onTime.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
            X_train_onTime = X_train_onTime.iloc[:, 1:]
            depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))
            depend_train = depend_train.drop(columns={'Unnamed: 0'})
        else:
            X_train_onTime = pd.read_csv("data/train_test_frames/final_train_x_onTimeDelivery.csv")
            X_train_onTime = X_train_onTime.drop(columns={'sellerId', 'orderDate', 'Unnamed: 0'})
            X_train_onTime = X_train_onTime.iloc[:, 1:]
            depend_train = pd.read_csv("data/train_test_frames/final_train_y_onTimeDelivery.csv")[criteria]
        depend_train = depend_train.astype(np.float32)

        if XGBoost:
            # Preparing test data
            depend_test = Y_test[criteria]
            depend_test = depend_test[XGB_pred_known == 1]
            X_test_XGB = X_test[XGB_pred_known == 1]

            # Predicting whether on time or not
            XGB2 = XGBmodel(X_train_onTime, X_test_XGB, depend_train, depend_test, criteria, balanced)
            print("XGB best parameters for predicting onTimeDelivery when predicted known:", XGB2.best_param)

            # Determining the best threshold
            if threshold:
                XGB_prob_onTime = XGB2.predp[:, 1]
                depend_test = depend_test.replace({'Unknown': 2})
                depend_test = depend_test.astype(np.float32)
                best_threshold = threshold_search(depend_test, XGB_prob_onTime, "XGB {0}".format(criteria), balanced)
                XGB_pred_onTime = np.ones(len(XGB_prob_onTime))
                XGB_pred_onTime[XGB_prob_onTime <= best_threshold] = 0

                # Saving new prediction
                save = pd.DataFrame(XGB_pred_onTime)
                save.to_csv("data/predictions/XGB_{0}_ct_prediction_{1}.csv".format(balanced, criteria))

            else:
                XGB_pred_onTime = XGB2.predc

            # Combining the two predictions
            final_pred_XGB = XGB_pred_known
            final_pred_XGB = final_pred_XGB.astype(object)
            final_pred_XGB[XGB_pred_known == 0] = 'Unknown'
            k = 0
            for j in range(0, len(final_pred_XGB)):
                if final_pred_XGB[j] == 1:
                    final_pred_XGB[j] = XGB_pred_onTime[k]
                    k = k + 1

            # Computing results & saving them
            depend_test = Y_test[criteria]
            classes = [0, 1, 'Unknown']
            print("XGB macro weighted F1 score for final {0} prediction: ".format(criteria),
                  macro_weighted_f1_print(depend_test, final_pred_XGB, classes))

            final_pred_XGB = pd.DataFrame(final_pred_XGB)
            final_pred_XGB.to_csv("data/predictions/XGB_{0}_final_prediction_{1}.csv".format(balanced, criteria))

        if NeuralNetwork:
            # Preparing test data
            depend_test = Y_test[criteria]
            depend_test = depend_test[NN_pred_known == 1]
            X_test_stand_NN = X_test_stand[NN_pred_known == 1]
            X_test_stand_NN = X_test_stand_NN.astype(np.float32)

            # Predicting whether on time or not
            X_train_stand = standardize_data(X_train_onTime).astype(np.float32)
            NN2 = NNmodel(X_train_stand, X_test_stand_NN, depend_train, depend_test, criteria, balanced)
            print("NN best parameters for predicting onTimeDelivery when predicted known:", NN2.best)

            # Determining the best threshold
            if threshold:
                NN_prob_onTime = NN2.predp
                NN_prob_onTime = NN_prob_onTime.T
                NN_prob_onTime = NN_prob_onTime[0]
                depend_test = depend_test.replace({'Unknown': 2})
                depend_test = depend_test.astype(np.float32)
                best_threshold = threshold_search(depend_test, NN_prob_onTime, "NN {0}".format(criteria), balanced)
                NN_pred_onTime = np.ones(len(NN_prob_onTime))
                NN_pred_onTime[NN_prob_onTime <= best_threshold] = 0

                # Saving new prediction
                save = pd.DataFrame(NN_pred_onTime)
                save.to_csv("data/predictions/NN_{0}_ct_prediction_{1}.csv".format(balanced, criteria))

            else:
                NN_pred_onTime = NN2.predc
                NN_pred_onTime = NN_pred_onTime.T
                NN_pred_onTime = NN_pred_onTime[0]
                print(NN_pred_onTime)

            # Combining the two predictions
            final_pred_NN = NN_pred_known
            final_pred_NN = final_pred_NN.astype(object)
            final_pred_NN[NN_pred_known == 0] = 'Unknown'
            m = 0
            for j in range(0, len(final_pred_NN)):
                if final_pred_NN[j] == 1:
                    final_pred_NN[j] = NN_pred_onTime[m]
                    m = m + 1

            # Computing results & saving them
            depend_test = Y_test[criteria]
            classes = [0, 1, 'Unknown']
            print("NN macro weighted F1 score for {0}: ".format(criteria),
                  macro_weighted_f1_print(depend_test, final_pred_NN, classes))

            final_pred_NN = pd.DataFrame(final_pred_NN)
            if threshold:
                final_pred_NN.to_csv("data/predictions/NN_{0}_final_ct_prediction_{1}.csv".format(balanced, criteria))
            else:
                final_pred_NN.to_csv("data/predictions/NN_{0}_final_c_prediction_{1}.csv".format(balanced, criteria))

    else:
        # Importing train data
        if balanced_data:
            X_train = pd.read_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))
            X_train = X_train.drop(columns={'Unnamed: 0', 'Unnamed: 0.1'})
            X_train = X_train.iloc[:, 1:]
            depend_train = pd.read_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))
            depend_train = depend_train.drop(columns={'Unnamed: 0'})
        else:
            depend_train = Y_train[criteria]
        depend_train = depend_train.astype(np.float32)
        depend_test = depend_test.astype(np.float32)

        # Predicting dependent variable
        if XGBoost:
            XGB = XGBmodel(X_train, X_test, depend_train, depend_test, criteria, balanced)
            print("XGB best parameters for {0}: ".format(criteria), XGB.best_param)
            print("XGB macro weighted F1 score for {0}: ".format(criteria), XGB.score)

            if threshold:
                # Determining the best threshold
                XGB_prob = XGB.predp[:, 1]
                best_threshold = threshold_search(depend_test, XGB_prob, "XGB {0}".format(criteria), balanced)
                XGB_pred = np.ones(len(XGB_prob))
                XGB_pred[XGB_prob < best_threshold] = 0
                print("Results after threshold optimization:")
                print("XGB macro weighted F1 score for {0} with optimized threshold: ".format(criteria),
                      macro_weighted_f1_print(depend_test, XGB_pred, [0, 1]))

                # Saving new prediction
                XGB_pred = pd.DataFrame(XGB_pred)
                XGB_pred.to_csv("data/predictions/XGB_{0}_ct_prediction_{1}.csv".format(balanced, criteria))

        if NeuralNetwork:
            X_train_stand = standardize_data(X_train).astype(np.float32)
            NN = NNmodel(X_train_stand, X_test_stand, depend_train, depend_test, criteria, balanced)
            print("NN best parameters for {0}: ".format(criteria), NN.best)
            print("NN macro weighted F1 score for {0}: ".format(criteria), NN.score)

            if threshold:
                # Determining the best threshold
                NN_prob = NN.predp
                NN_prob = NN_prob.T
                NN_prob = NN_prob[0]
                best_threshold = threshold_search(depend_test, NN_prob, "NN {0}".format(criteria), balanced)
                NN_pred = np.ones(len(NN_prob))
                NN_pred[NN_prob < best_threshold] = 0
                print("***RESULTS WITH THRESHOLD***")
                print("NN macro weighted F1 score for {0} with optimized threshold:".format(criteria),
                      macro_weighted_f1_print(depend_test, NN_pred, [0, 1]))

                # Saving new prediction
                NN_pred = pd.DataFrame(NN_pred)
                NN_pred.to_csv("data/predictions/NN_{0}_ct_prediction_{1}.csv".format(balanced, criteria))

# Classifying the orders based on their predictions
if XGBoost:
    match_classification("XGB", balanced_data, threshold)

if NeuralNetwork:
    match_classification("NN", balanced_data, threshold)
