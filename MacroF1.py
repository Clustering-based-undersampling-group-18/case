"""
This script contains functions to calculate prediction metrics and optimize the threshold
These functions are used in MainModels.py
"""

import numpy as np
import pandas as pd
from networkx.drawing.tests.test_pylab import plt
from sklearn.model_selection import train_test_split


# Function to compute the macro_weighted F1 score of a prediction
def macro_weighted_f1(true, predict, classes):
    macro_f1 = 0
    precision = 0
    recall = 0
    if 'Unknown' in classes:
        true = true.replace({'Unknown': 2})
        predict[predict == 'Unknown'] = 2
        classes = [0, 1, 2]

    true = true.astype(np.float32)
    predict = predict.astype(np.float32)
    for c in classes:
        # correctly predicted
        true_positives_c = 0
        for i in range(0, len(true)):
            if true[i] == c:
                if predict[i] == c:
                    true_positives_c += 1

        false_positives_c = 0
        for i in range(0, len(true)):
            if true[i] != c:
                if predict[i] == c:
                    false_positives_c += 1

        if (true_positives_c + false_positives_c) == 0:
            precision_c = 0.0
        else:
            precision_c = np.divide(true_positives_c, (true_positives_c + false_positives_c))

        precision += precision_c/len(classes)

        false_negatives_c = 0
        for i in range(0, len(true)):
            if true[i] == c:
                if predict[i] != c:
                    false_negatives_c += 1

        if (true_positives_c + false_negatives_c) == 0:
            recall_c = 0.0
        else:
            recall_c = np.divide(true_positives_c, (true_positives_c + false_negatives_c))

        recall += recall_c/len(classes)

        if (precision_c+recall_c) == 0:
            f1_c = 0
        else:
            f1_c = np.divide((2*precision_c*recall_c), (precision_c+recall_c))
        macro_f1 += np.divide(1, len(classes)) * f1_c

    return macro_f1


# Function to compute the macro_weighted F1 score of a prediction with printing
def macro_weighted_f1_print(true, predict, classes):
    macro_f1 = 0
    precision = 0
    recall = 0
    if 'Unknown' in classes:
        true = true.replace({'Unknown': 2})
        predict[predict == 'Unknown'] = 2
        classes = [0, 1, 2]

    true = true.astype(np.float32)
    predict = predict.astype(np.float32)
    for c in classes:
        # correctly predicted
        true_positives_c = 0
        for i in range(0, len(true)):
            if true[i] == c:
                if predict[i] == c:
                    true_positives_c += 1

        false_positives_c = 0
        for i in range(0, len(true)):
            if true[i] != c:
                if predict[i] == c:
                    false_positives_c += 1

        if (true_positives_c + false_positives_c) == 0:
            precision_c = 0.0
        else:
            precision_c = np.divide(true_positives_c, (true_positives_c + false_positives_c))
        print("Precision {0}: ".format(c), precision_c)
        precision += precision_c/len(classes)

        false_negatives_c = 0
        for i in range(0, len(true)):
            if true[i] == c:
                if predict[i] != c:
                    false_negatives_c += 1

        if (true_positives_c + false_negatives_c) == 0:
            recall_c = 0.0
        else:
            recall_c = np.divide(true_positives_c, (true_positives_c + false_negatives_c))

        print("Recall {0}: ".format(c), recall_c)
        recall += recall_c/len(classes)

        if (precision_c + recall_c) == 0:
            f1_c = 0
        else:
            f1_c = np.divide((2 * precision_c * recall_c), (precision_c + recall_c))
        print("F1 {0}: ".format(c), f1_c)
        macro_f1 += np.divide(1, len(classes)) * f1_c

    print("Macro recall:", recall)
    print("Macro precision:", precision)
    return macro_f1


# Function that optimizes the prediction threshold
def threshold_search(true, prob, criteria):
    true = true.to_numpy()
    prob_train, prob_test, true_train, true_test = train_test_split(prob, true, test_size=0.2, random_state=1234)

    thresholds = np.linspace(0, 1, 101)
    all_f1_train = np.zeros(len(thresholds))
    for j in range(len(thresholds)):
        predictions_train = np.ones(len(prob_train))
        predictions_train[prob_train < thresholds[j]] = 0

        macro_f1_train = macro_weighted_f1(true_train, predictions_train, [0, 1])
        all_f1_train[j] = macro_f1_train

    best_threshold = thresholds[np.where(max(all_f1_train) == all_f1_train)]
    best_threshold = best_threshold[0]
    predictions_test = np.ones(len(prob_test))
    predictions_test[prob_test < best_threshold] = 0
    print("The best threshold for this prediction is: %s" % best_threshold)

    plt.plot(thresholds, all_f1_train, 'b')
    plt.axvline(x=0.5, linestyle=':', color='r')
    plt.axvline(x=best_threshold, linestyle='--', color='g')
    plt.axhline(y=all_f1_train[np.where(thresholds == 0.5)], linestyle=':', color='r')
    plt.axhline(y=max(all_f1_train), linestyle='--', color='g')
    plt.xlabel("Threshold")
    plt.ylabel("Macro F1")
    plt.savefig('{0} threshold plot.png'.format(criteria), bbox_inches='tight')

    return best_threshold


# This function performs the match classification based on the business decision tree + the criteria predictions
def match_classification():
    XGBoost = True
    NeuralNetwork = True
    threshold = True


    true_values = pd.read_csv("data/train_test_frames/final_test_x.csv")["detailedMatchClassification"]
    predictions = pd.Series(["KNOWN HAPPY"] * len(true_values))

    if not threshold:
        """ 
        late_predictions = pd.read_csv("data/results/Imbalanced/NN_onTimeDelivery_HeleData.csv", header=None)[0]
        print(late_predictions)
        indices_late_prediction = np.where(late_predictions == 0)[0]
        predictions.loc[indices_late_prediction] = "UNHAPPY"

        unknown_predictions = pd.read_csv("data/results/Imbalanced/NN_DeliveryKnownUnknown_HeleData.csv", header=None)[0]
        indices_unknown_prediction = np.where(unknown_predictions == 0)[0]
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"
        """
        times = pd.read_csv("data/predictions/XGB_balanced_final_prediction_onTimeDelivery.csv", header=None,
                            skiprows=1)[1]
        indices_late_prediction = np.where(times == 0)[0]
        predictions.loc[indices_late_prediction] = "UNHAPPY"
        indices_unknown_prediction = np.where(times == 2)[0]
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"

    else:
        total_time = pd.read_csv("data/predictions/Im_NN_totalTime_after_threshold.csv", skiprows=1,
                                 header=None)[1]

        indices_unknown_prediction = np.where(total_time == 2)[0]
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"

        indices_late_prediction = np.where(total_time == 0)[0]
        predictions.loc[indices_late_prediction] = "UNHAPPY"

        print(predictions.value_counts())

    cancel_predictions = pd.read_csv("data/results/Imbalanced_threshold/Im_NN_noCancellation_after_threshold.csv", skiprows=1, header=None)[1]
    indices_cancel_prediction = np.where(cancel_predictions == 0)[0]
    predictions.loc[indices_cancel_prediction] = "UNHAPPY"

    return_predictions = pd.read_csv("data/results/Imbalanced_threshold/Im_NN_noReturn_after_threshold.csv", skiprows=1, header=None)[1]
    indices_return_prediction = np.where(return_predictions == 0)[0]
    predictions.loc[indices_return_prediction] = "UNHAPPY"

    case_predictions = pd.read_csv("data/results/Imbalanced_threshold/Im_NN_noCase_after_threshold.csv", skiprows=1, header=None)[1]
    indices_case_prediction = np.where(case_predictions == 0)[0]
    predictions.loc[indices_case_prediction] = "UNHAPPY"

    predictions_train, predictions, true_values_train, true_values = train_test_split(predictions, true_values, test_size=0.2, random_state=1234)

    print(predictions.value_counts())
    print(true_values.value_counts())

    true_values = true_values.to_numpy()
    predictions = predictions.to_numpy()

    correct_minority = 0
    for i in range(0, len(true_values)):
        if predictions[i] == 0:
            if true_values[i] == 0:
                correct_minority += 1

    classes = ["UNKNOWN", "KNOWN HAPPY", "UNHAPPY"]
    macro_weighted_f1_print(true_values, predictions, classes)

