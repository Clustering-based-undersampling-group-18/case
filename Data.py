import pandas as pd
from RandomForest import RandomForest
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
           'frequencySeller', 'dayOfTheWeek', 'monthOfTheYear']]
X = pd.get_dummies(X, columns=['countryCode', 'fulfilmentType', 'productGroup', 'countryOriginSeller',
                               'currentCountryAvailabilitySeller', 'dayOfTheWeek', 'monthOfTheYear'])
features = list(X.columns)
X = X.to_numpy()

for depend in Y.T:
    depend = depend.T
    RandomForest(X, Y)


