import numpy as np


def macro_weighted_f1(true, predict, classes):
    macro_f1 = 0
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
            if true[i] == c:
                if predict[i] != c:
                    false_positives_c += 1

        precision_c = np.divide(true_positives_c, (true_positives_c + false_positives_c))

        false_negatives_c = 0
        for i in range(0, len(true)):
            if true[i] != c:
                if predict[i] == c:
                    false_negatives_c += 1

        recall_c = np.divide(true_positives_c, (true_positives_c + false_negatives_c))
        f1_c = np.divide((2*precision_c*recall_c), (precision_c+recall_c))
        macro_f1 += np.divide(1, len(classes)) * f1_c

    return macro_f1
