import pandas as pd
from scipy import stats
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
import pydot
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

    # Splitting the data
    X_train, X_test, Y_train, Y_test = train_test_split(X, depend, test_size=0.3, random_state=1234)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=1234)

    # Hyperparameter sets
    hyper_param = {'n_estimators': stats.randint(150, 1000),
                   'learning_rate': stats.uniform(0.01, 0.6),
                   'subsample': stats.uniform(0.3, 0.9),
                   'max_depth': [3, 4, 5, 6, 7, 8, 9],
                   'colsample_bytree': stats.uniform(0.5, 0.9),
                   'min_child_weight': [1, 2, 3, 4]}

    # Grid search for the hyperparameters
    RF = XGBClassifier(use_label_encoder=False)
    cv = KFold(n_splits=5, random_state=1)
    RSCV = RandomizedSearchCV(RF, param_distributions=hyper_param, cv=cv, n_iter=5, scoring='roc_auc', error_score=0,
                              verbose=3, n_jobs=-1)
    random_search = RSCV.fit(X_val, Y_val)

    # Prints the best hyperparameters
    print("Best hyperparameter values:", random_search.best_params_)

    # Predicts the test set using the best model from the grid search
    RF = XGBClassifier(**random_search.best_params_)
    RF.fit(X_train, Y_train)
    score = RF.score(X_test, Y_test)

    # Prints the accuracy of the prediction
    print("RF prediction ROC AUC:", score)

# Visualizes the fifth tree
tree = RF.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features, rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
