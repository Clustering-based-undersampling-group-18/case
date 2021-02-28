import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        if true[i] == c:
            if predict[i] != c:
                false_positives_c += 1
    precision_c = np.divide(true_positives_c, (true_positives_c + false_positives_c))

    false_negatives_c = 0
    for i in range(0, len(true)):
        if true[i] != c:
            if predict[i] == c:
                false_negatives_c += 1

    if (true_positives_c + false_negatives_c) == 0:
        recall_c = 0.0
    else:
        recall_c = np.divide(true_positives_c, (true_positives_c + false_negatives_c))

    return precision_c, recall_c


def f1(precision, recall):
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = np.divide((2 * precision * recall), (precision + recall))
    return f1


def run():
    true_values = pd.read_csv("data/final_test_y.csv")["noCase"]
    true_values = true_values.to_numpy()
    probabilities = \
        pd.read_csv("data/predictedP_NN_noCase.csv", header=None)[0]

    probabilities_train, probabilities_test, true_values_train, true_values_test = \
        train_test_split(probabilities, true_values, test_size=0.2, random_state=1234)

    thresholds = np.linspace(0, 1, 101)
    all_macrof1_train = np.zeros(len(thresholds))
    all_macrof1_test = np.zeros(len(thresholds))
    for j in range(len(thresholds)):
        predictions_train = np.ones(len(probabilities_train))
        predictions_test = np.ones(len(probabilities_test))

        predictions_train[probabilities_train <= thresholds[j]] = 0
        predictions_test[probabilities_test <= thresholds[j]] = 0

        macro_f1_train = 0
        macro_f1_test = 0
        classes = [0, 1]
        for c in classes:
            precision_c_train, recall_c_train = precision_and_recall_c(c, true_values_train, predictions_train)
            precision_c_test, recall_c_test = precision_and_recall_c(c, true_values_test, predictions_test)
            f1_c_train = f1(precision_c_train, recall_c_train)
            f1_c_test = f1(precision_c_test, recall_c_test)
            macro_f1_train += np.divide(1, len(classes)) * f1_c_train
            macro_f1_test += np.divide(1, len(classes)) * f1_c_test

        all_macrof1_train[j] = macro_f1_train
        all_macrof1_test[j] = macro_f1_test
    return all_macrof1_train, all_macrof1_test


def run_test(best_threshold):
    true_values = pd.read_csv("data/final_test_y.csv")["noCase"]
    true_values = true_values.to_numpy()
    probabilities = \
        pd.read_csv("data/predictedP_NN_noCase.csv", header=None)[0]

    probabilities_train, probabilities_test, true_values_train, true_values_test = \
        train_test_split(probabilities, true_values, test_size=0.2, random_state=1234)

    predictions = np.ones(len(probabilities_test))
    predictions[probabilities_test <= best_threshold[0]] = 0

    true_values = true_values_test

    correct_minority = 0
    for i in range(0, len(true_values)):
        if predictions[i] == 0:
            if true_values[i] == 0:
                correct_minority += 1

    macro_f1 = 0
    macro_recall = 0
    macro_precision = 0
    classes = [0, 1]
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


[all_macrof1_train, all_macrof1_test] = run()
threshold = np.linspace(0, 1, 101)
plt.plot(threshold, all_macrof1_train, 'b')
plt.axvline(x=0.5, linestyle='--', color='r')
plt.axvline(x=threshold[np.where(max(all_macrof1_train) == all_macrof1_train)], linestyle='--', color='g')
plt.axhline(y=all_macrof1_train[np.where(threshold == 0.5)], linestyle='--', color='r')
plt.axhline(y=max(all_macrof1_train), linestyle='--', color='g')
plt.xlabel("threshold")
plt.ylabel("macro F1")
plt.show()

best_threshold = threshold[np.where(max(all_macrof1_train) == all_macrof1_train)]
print("The best threshold is: %s" % best_threshold)

run_test(best_threshold)
