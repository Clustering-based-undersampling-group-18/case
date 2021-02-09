import pandas as pd
from XGBoost import RandomForest
from sklearn.model_selection import train_test_split
import random

# Importing data
random.seed(1234)
missing_value_formats = ["n.a.", "?", "NA", "n/a", "na", "--", "NaN", " ", ""]
frame = pd.read_csv("data/frame.csv", na_values=missing_value_formats,
                    dtype={'onTimeDelivery': str, 'datetTimeFirstDeliveryMoment': object, 'returnCode': object,
                           'transporterNameOther': object, 'cancellationReasonCode': object})  # 2110338

# Preparing dependent variables
Y = frame[['noCancellation', 'onTimeDelivery', 'noReturn', 'noCase']]
Y = pd.get_dummies(Y, columns=['onTimeDelivery'])
Y = Y.replace(to_replace=True, value=1)
Y = Y.replace(to_replace=False, value=0)
Y = Y.to_numpy()

# Preparing explanatory variables
X = frame[['totalPrice', 'quantityOrdered', 'countryCode', 'fulfilmentType', 'promisedDeliveryDate',
           'productGroup', 'registrationDateSeller', 'countryOriginSeller', 'currentCountryAvailabilitySeller',
           'day_of_week', 'month_of_year']]
X = pd.get_dummies(X, columns=['countryCode', 'fulfilmentType', 'productGroup', 'countryOriginSeller',
                               'currentCountryAvailabilitySeller', 'day_of_week', 'month_of_year'])
features = list(X.columns)
X = X.to_numpy()

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

# Predicting dependent variable with XGBoost Random Forest
for depend in Y.T:
    depend = depend.T
    RF = RandomForest(X, depend)
    print("RF best parameters: ", RF.best_param)
    print("RF prediction accuracy: ", RF.score)

