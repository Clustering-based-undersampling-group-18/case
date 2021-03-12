import pickle
import random
import math
import pandas as pd
import numpy as np
from scipy.spatial import distance

variable = "OnTime"

if variable == "OnTime":
    probabilities_time_test = pd.read_csv("data/results/Imbalanced/NN_Probabilities_onTimeDelivery_HeleData.csv", header=None)[0]
    print(probabilities_time_test)
    probabilities_known_test = pd.read_csv("data/results/Imbalanced/NN_Prob_DeliveryKnownUnknown_HeleData.csv", header=None)[0]
    predictions = np.ones(len(probabilities_known_test))
    predictions[probabilities_time_test <= 0.9] = 0
    predictions[probabilities_known_test <= 0.52] = 2

    predictions = pd.DataFrame(predictions)
    print(predictions[0].value_counts())
    predictions = predictions.astype(int)

    predictions.to_csv("data/results/Imbalanced_threshold/Im_NN_totalTime_after_threshold.csv", header=[0])
else:
    data = pd.read_csv("data/results/Imbalanced/NN_Probabilities_noReturn2_HeleData.csv", header=None, skiprows=1)[0]
    print(data[data<0.89])
    best_threshold = 0.89
    data[data < best_threshold] = int(0)
    data[data >= best_threshold] = int(1)
    data.to_csv("data/results/Imbalanced_threshold/Im_NN_noReturn_after_threshold.csv", header=[0])


