import numpy as np
from networkx.drawing.tests.test_pylab import plt
from sklearn.model_selection import train_test_split


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

        precision_c = np.divide(true_positives_c, (true_positives_c + false_positives_c))
        precision += precision_c/len(classes)

        false_negatives_c = 0
        for i in range(0, len(true)):
            if true[i] == c:
                if predict[i] != c:
                    false_negatives_c += 1

        recall_c = np.divide(true_positives_c, (true_positives_c + false_negatives_c))
        recall += recall_c/len(classes)

        f1_c = np.divide((2*precision_c*recall_c), (precision_c+recall_c))
        macro_f1 += np.divide(1, len(classes)) * f1_c

    print("Recall:", recall)
    print("Precision:", precision)
    return macro_f1


def threshold_search(true, prob):
    true = true.to_numpy()
    prob_train, prob_test, true_train, true_test = train_test_split(prob, true, test_size=0.2, random_state=1234)

    thresholds = np.linspace(0.1, 0.9, 99)
    all_f1_train = np.zeros(len(thresholds))
    all_f1_test = np.zeros(len(thresholds))
    for j in range(len(thresholds)):
        predictions_train = np.ones(len(prob_train))
        predictions_test = np.ones(len(prob_test))

        predictions_train[prob_train <= thresholds[j]] = 0
        predictions_test[prob_test <= thresholds[j]] = 0

        macro_f1_train = macro_weighted_f1(true_train, predictions_train, [0, 1])
        macro_f1_test = macro_weighted_f1(true_test, predictions_test, [0, 1])

        all_f1_train[j] = macro_f1_train
        all_f1_test[j] = macro_f1_test

    best_threshold = thresholds[np.where(max(all_f1_train) == all_f1_train)]
    best_threshold = best_threshold[0]
    print("The best threshold is: %s" % best_threshold)

    plt.plot(thresholds, all_f1_train, 'b')
    plt.axvline(x=0.5, linestyle='--', color='r')
    plt.axvline(x=best_threshold, linestyle='--', color='g')
    plt.axhline(y=all_f1_train[np.where(thresholds == 0.5)], linestyle='--', color='r')
    plt.axhline(y=max(all_f1_train), linestyle='--', color='g')
    plt.xlabel("Threshold")
    plt.ylabel("Macro F1")
    plt.savefig('demo.png', bbox_inches='tight')
    return best_threshold
