import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def precision_and_recall_c(c, true, predict):
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

    false_negatives_c = 0
    for i in range(0, len(true)):
        if true[i] == c:
            if predict[i] != c:
                false_negatives_c += 1

    if (true_positives_c + false_negatives_c) == 0:
        recall_c = 0.0
    else:
        recall_c = np.divide(true_positives_c, (true_positives_c + false_negatives_c))

    return precision_c, recall_c, true_positives_c


def f1(precision, recall):
    if precision==0 and recall ==0:
        return 0
    else:
        return np.divide((2 * precision * recall), (precision + recall))


def run():
    Criteria = "anders"
    #true_values = pd.read_csv("data/train_test_frames/final_test_y_Unknown.csv")["onTimeDelivery"]
    #true_values = true_values.replace("Unknown", 2).astype(np.float32)
    true_values = pd.read_csv("data/train_test_frames/final_test_y_Unknown")["onTimeDelivery"]
    #predictions = pd.read_csv("data/Imbalanced/XGB_imbalanced_p_prediction_Unknown.csv")["1"].astype(np.float32)
    predictions = pd.read_csv("data/Imbalanced/XGB_imbalanced_p_prediction_Unknown.csv")["1"].astype(np.float32)
    #predictions = pd.read_csv("data/Imbalanced/Balanced/XGB_balanced_p_prediction_Unknown.csv", skiprows=1, header=None)[1].astype(np.float32)
    print(predictions)

    if Criteria == "onTimeDeliveryImbalanced":
        predictions_unknown = pd.read_csv("data/Imbalanced/NN_DeliveryKnownUnknown_HeleData.csv", header=None)
        predictions_ontime = pd.read_csv("data/Imbalanced/NN_onTimeDelivery_HeleData.csv", header=None)
        tst1 = np.where(predictions_unknown[0]==0)[0]
        tst2 = np.where(predictions_ontime[0]==0)[0]
        predictions = predictions_unknown
        predictions = predictions.replace(0, 2).astype(np.float32)
        indices_knowns = np.where(predictions == np.float(1))[0]
        predictions.loc[indices_knowns, 0] = predictions_ontime[0]
        predictions = predictions[0].astype(np.float32)

    if Criteria == "onTimeDelivery":
        predictions_unknown = pd.read_csv("data/Imbalanced/Balanced/predictedY_NN_Unknown.csv", header=None)
        predictions_ontime = pd.read_csv("data/Imbalanced/Balanced/predictedY_NN_onTimeDelivery.csv", header=None)[0]
        predictions = predictions_unknown
        predictions = predictions.replace(float(0), float(2))
        indices_knowns = np.where(predictions != np.float(2))[0]
        predictions.loc[indices_knowns, 0] = predictions_ontime
        predictions = predictions[0].astype(np.float32)
        predictions = pd.read_csv("data/Imbalanced/Balanced/XGB_balanced_combined_c_prediction_onTimeDelivery.csv", skiprows=1, header= None)[1].astype(np.float32)
        print(predictions)

    if Criteria == "check":
        to_keep = np.where(true_values != 2)[0]
        true_values = true_values.loc[to_keep, ]
        print(true_values)
        predictions_ontime = pd.read_csv("data/Imbalanced/Balanced/XGB_balanced_c_prediction_onTimeDelivery.csv", skiprows=1,  header=None)
        print(predictions_ontime)
        predictions = predictions_ontime.loc[to_keep, 1]
        print(predictions)

    classes = true_values.unique()

    predictions[predictions >= 0.59] = int(1)
    predictions[predictions<0.59] = int(0)
    print(true_values.value_counts())
    print(predictions.value_counts())
    true_values = true_values.to_numpy()
    predictions = predictions.to_numpy()

    correct_minority = 0
    for i in range(0, len(true_values)):
        if predictions[i] == 0:
            if true_values[i] == 0:
                correct_minority += 1

    macro_f1 = 0
    macro_recall = 0
    macro_precision = 0

    for c in classes:
        precision_c, recall_c = precision_and_recall_c(c, true_values, predictions)
        print("Precision class {0}:".format(c), precision_c)
        print("Recall class {0}:".format(c), recall_c)
        macro_precision += np.divide(1, len(classes)) * precision_c
        macro_recall += np.divide(1, len(classes)) * recall_c
        print("F1 class {0}:".format(c), f1(precision_c, recall_c))
        f1_c = f1(precision_c, recall_c)
        macro_f1 += np.divide(1, len(classes)) * f1_c

    print(macro_f1, macro_precision, macro_recall)
    print("Macro precision:", macro_precision)
    print("Macro recall:", macro_recall)
    print("Macro F1", macro_f1)


def both_models_both_true():
    true_values = pd.read_csv("data/train_test_frames/final_test_y.csv")["onTimeDelivery"]
    true_values = true_values.replace(to_replace="0", value=np.float(1))
    true_values = true_values.replace(to_replace=0, value=np.float(1))
    true_values = true_values.replace(to_replace="1", value=np.float(1))
    true_values = true_values.replace(to_replace=1, value=np.float(1))
    true_values = true_values.replace(to_replace="Unknown", value=np.float(0))
    true_values.to_csv("data/train_test_frames/final_test_y_Unknown.csv")

    print(true_values)
    temp = pd.read_csv("data/Imbalanced/NN_DeliveryKnownUnknown_HeleData.csv", header=None, names=["test"])["test"]
    # indices_for_1 = np.where(temp>0.39)[0]
    # indices_for_0 = np.where(temp<=0.39)[0]
    temp[temp >= 0.39] = int(1)
    temp[temp < 0.39] = int(0)
    temp.to_csv("test_ding_theshold.csv")


def match_classification(model, balanced, threshold):
    threshold_boolean = True

    true_values = pd.read_csv("data/train_test_frames/final_test_x.csv")["detailedMatchClassification"]
    predictions = pd.Series(["KNOWN HAPPY"] * len(true_values))

    if not threshold_boolean:
        """ 
        late_predictions = pd.read_csv("data/results/Imbalanced/NN_onTimeDelivery_HeleData.csv", header=None)[0]
        print(late_predictions)
        indices_late_prediction = np.where(late_predictions == 0)[0]
        predictions.loc[indices_late_prediction] = "UNHAPPY"

        unknown_predictions = pd.read_csv("data/results/Imbalanced/NN_DeliveryKnownUnknown_HeleData.csv", header=None)[0]
        indices_unknown_prediction = np.where(unknown_predictions == 0)[0]
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"
        """
        times = pd.read_csv("data/results/Balanced/XGB_balanced_final_prediction_onTimeDelivery.csv", header=None, skiprows=1)[1]
        print(times)
        print(times.value_counts())
        indices_late_prediction = np.where(times == 0)[0]
        print(indices_late_prediction)
        predictions.loc[indices_late_prediction] = "UNHAPPY"
        indices_unknown_prediction = np.where(times == 2)[0]
        print(indices_unknown_prediction)
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"

    else:
        total_time = pd.read_csv("data/results/Imbalanced_threshold/Im_NN_totalTime_after_threshold.csv", skiprows=1, header=None)[1]

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

    macro_f1 = 0
    macro_recall = 0
    macro_precision = 0
    correct = 0
    classes = ["UNKNOWN", "KNOWN HAPPY", "UNHAPPY"]
    for c in classes:
        precision_c, recall_c, tp_c = precision_and_recall_c(c, true_values, predictions)
        correct += tp_c
        print("Precision class {0}:".format(c), precision_c)
        print("Recall class {0}:".format(c), recall_c)
        macro_precision += np.divide(1, len(classes)) * precision_c
        macro_recall += np.divide(1, len(classes)) * recall_c
        print("F1 class {0}:".format(c), f1(precision_c, recall_c))
        f1_c = f1(precision_c, recall_c)
        macro_f1 += np.divide(1, len(classes)) * f1_c

    total = len(true_values)
    accuracy = np.divide(correct, total)
    print(macro_f1, macro_precision, macro_recall)
    print("Macro precision:", macro_precision)
    print("Macro recall:", macro_recall)
    print("Macro F1:", macro_f1)
    print("Accuracy:", accuracy )


def match_classification_xgb():
    threshold_boolean = True

    true_values = pd.read_csv("data/train_test_frames/final_test_x.csv")["detailedMatchClassification"]
    predictions = pd.Series(["KNOWN HAPPY"] * len(true_values))

    cancel_predictions = pd.read_csv("data/results/Balanced_threshold/XGB_noCancellation_after_threshold.csv", header=None, skiprows=1)[1]
    indices_cancel_prediction = np.where(cancel_predictions == 0)[0]
    predictions.loc[indices_cancel_prediction] = "UNHAPPY"

    return_predictions = pd.read_csv("data/results/Balanced_threshold/XGB_noReturn_after_threshold.csv", header=None, skiprows=1)[1]
    indices_return_prediction = np.where(return_predictions == 0)[0]
    predictions.loc[indices_return_prediction] = "UNHAPPY"

    case_predictions = pd.read_csv("data/results/Balanced_threshold/XGB_noCase_after_threshold.csv", header=None, skiprows=1)[1]
    indices_case_prediction = np.where(case_predictions == 0)[0]
    predictions.loc[indices_case_prediction] = "UNHAPPY"

    if not threshold_boolean:
        """ 
        late_predictions = pd.read_csv("data/results/Imbalanced/NN_onTimeDelivery_HeleData.csv", header=None)[0]
        print(late_predictions)
        indices_late_prediction = np.where(late_predictions == 0)[0]
        predictions.loc[indices_late_prediction] = "UNHAPPY"

        unknown_predictions = pd.read_csv("data/results/Imbalanced/NN_DeliveryKnownUnknown_HeleData.csv", header=None)[0]
        indices_unknown_prediction = np.where(unknown_predictions == 0)[0]
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"
        """
        times = pd.read_csv("data/results/Balanced/XGB_balanced_final_prediction_onTimeDelivery.csv", header=None, skiprows=1)[1]
        print(times)
        print(times.value_counts())
        indices_late_prediction = np.where(times == 0)[0]
        print(indices_late_prediction)
        predictions.loc[indices_late_prediction] = "UNHAPPY"
        indices_unknown_prediction = np.where(times == 2)[0]
        print(indices_unknown_prediction)
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"

    else:
        total_time = pd.read_csv("data/results/Balanced_threshold/XGB_balanced_final_prediction_onTimeDelivery_goede.csv", header=None, skiprows=1)[1]
        print(total_time)
        indices_late_prediction = np.where(total_time == 0)[0]
        predictions.loc[indices_late_prediction] = "UNHAPPY"

        indices_unknown_prediction = np.where(total_time == 2)[0]
        predictions.loc[indices_unknown_prediction] = "UNKNOWN"

        indices_happy_prediction = np.where(total_time == 1)[0]
        predictions.loc[indices_happy_prediction] = "KNOWN HAPPY"

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

    macro_f1 = 0
    macro_recall = 0
    macro_precision = 0
    correct = 0
    classes = ["UNKNOWN", "KNOWN HAPPY", "UNHAPPY"]
    for c in classes:
        precision_c, recall_c, tp_c = precision_and_recall_c(c, true_values, predictions)
        correct += tp_c
        print("Precision class {0}:".format(c), precision_c)
        print("Recall class {0}:".format(c), recall_c)
        macro_precision += np.divide(1, len(classes)) * precision_c
        macro_recall += np.divide(1, len(classes)) * recall_c
        print("F1 class {0}:".format(c), f1(precision_c, recall_c))
        f1_c = f1(precision_c, recall_c)
        macro_f1 += np.divide(1, len(classes)) * f1_c

    total = len(true_values)
    accuracy = np.divide(correct, total)
    print(macro_f1, macro_precision, macro_recall)
    print("Macro precision:", macro_precision)
    print("Macro recall:", macro_recall)
    print("Macro F1:", macro_f1)
    print("Accuracy:", accuracy )


match_classification()