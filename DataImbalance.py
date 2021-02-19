# load modules and packages
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import pickle
import logging
import time


def load_data():
    """ This function loads and prepares the data """
    frame = pd.read_csv("data/frame.csv",
                        dtype={'onTimeDelivery': str, 'datetTimeFirstDeliveryMoment': object, 'returnCode': object,
                               'transporterNameOther': object,
                               'cancellationReasonCode': object})

    frame_X = frame[['sellerId', 'orderDate', 'countryCode', 'fulfilmentType', 'productGroup', 'countryOriginSeller',
                     'currentCountryAvailabilitySeller', 'totalPrice', 'quantityOrdered', 'promisedDeliveryDate',
                     'registrationDateSeller', 'day_of_week', 'month_of_year']]

    X = frame_X

    categorical_variables = ['countryCode', 'fulfilmentType', 'productGroup', 'countryOriginSeller',
                             'currentCountryAvailabilitySeller', 'day_of_week', 'month_of_year']
    X_frame = pd.get_dummies(X, columns=categorical_variables)

    frame_Y = frame[['noCancellation', 'onTimeDelivery', 'noReturn', 'noCase']]

    Y = frame_Y

    Y = Y.replace(to_replace=True, value=np.float(1))
    Y = Y.replace(to_replace=False, value=np.float(0))
    Y = Y.replace(to_replace="true", value=np.float(1))
    Y_frame = Y.replace(to_replace="false", value=np.float(0))
    return X_frame, Y_frame


def five_fold_cv(X):
    """ This function splits the input data in 5 training sets with corresponding validation sets """
    np.random.seed(1234)
    kf = KFold(n_splits=5, shuffle=True)

    train_indices = []
    val_indices = []
    for train_index, val_index in kf.split(X):
        train_indices.append(train_index)
        val_indices.append(val_index)

    with open('data/train_indices.pkl', 'wb') as f:
        pickle.dump(train_indices, f)
    f.close()

    with open('data/val_indices.pkl', 'wb') as f:
        pickle.dump(val_indices, f)
    f.close()
    return train_indices, val_indices


def frequency_seller(train_data, test_data):
    """ This function adds a new column containing the total amount of orders a seller receives"""
    # compute frequency for training, add those number to test set (if seller not known in train then add 0 for test)
    temp_count_dict = pd.Series.to_dict(train_data.groupby('sellerId')['orderDate'].nunique())
    train_data.insert(loc=len(train_data.columns), column="frequencySeller",
                      value=train_data['sellerId'].map(temp_count_dict))
    test_data.insert(loc=len(test_data.columns), column="frequencySeller",
                     value=test_data['sellerId'].map(temp_count_dict))

    return train_data, test_data


def standardize_data(X_train):
    """ This function standardized/ normalizes the data, required for the KMeans algorithm"""
    columns_to_standardize = ['totalPrice', 'quantityOrdered', 'promisedDeliveryDate', 'registrationDateSeller',
                              'frequencySeller']

    data_to_standardize = X_train[columns_to_standardize]

    scaler = StandardScaler().fit(data_to_standardize)
    standardized_data = X_train.copy()
    standardized_columns = scaler.transform(data_to_standardize)
    standardized_data[columns_to_standardize] = standardized_columns
    return standardized_data


def k_means_plus_two_strategies(standardized_data_x, data_y, column_name, normal_data_x):
    """ This function performs KMeans algorithm and updates the training set according to the two strategies"""
    # step 5.1: count the occurrences where column condition is not met  (e.g. number of returns)
    criteria_data = data_y[[column_name]]
    minority_class = criteria_data[criteria_data[column_name] == np.float(0)]
    majority_class = criteria_data[criteria_data[column_name] == np.float(1)]
    majority_rows = list(majority_class.index.values)
    minority_rows = list(minority_class.index.values)

    print("Size of majority class:", len(majority_rows))
    print("Size of minority class:", len(minority_rows))

    # step 5.2: perform kmeans clustering
    N = len(minority_class)  # = number of minority class variables
    if N == 0:
        logging.warning("There are no observations in the minority class!")
        empty_frame = pd.DataFrame([])
        empty_series = pd.Series([])
        return empty_frame, empty_frame, empty_series

    print("Seconds passed before starting KMeans:", time.perf_counter())
    n_c = 10000
    majority_data_stdz = standardized_data_x.loc[majority_rows, :]
    majority_data = normal_data_x.loc[majority_rows, :]
    kmeans = MiniBatchKMeans(n_clusters=n_c, batch_size=N).fit(majority_data_stdz)
    print("Seconds passed when KMeans completed:", time.perf_counter())
    majority_data['cluster'] = kmeans.predict(majority_data_stdz)
    print("Seconds passed when prediction completed:", time.perf_counter())

    # step 5.2.2: print some count information about the clusters
    # summary = majority_data.groupby(['cluster']).mean()
    # summary['count'] = majority_data['cluster'].value_counts()
    # summary = summary.sort_values(by='count', ascending=False)

    # step 5.3: find the new values
    M = len(majority_rows)
    minority_data = normal_data_x.loc[minority_rows, :]
    new_X = minority_data

    count = 0
    for c in range(0, n_c):
        data = majority_data[majority_data["cluster"] == c]
        data = data.drop(columns={'cluster'})
        obs_from_c = round((len(data) / M) * N)
        count = count + obs_from_c
        random_majors_in_c = data.sample(n=obs_from_c, random_state=1234)
        new_X = new_X.append(random_majors_in_c, ignore_index=True)
    print("Clustering succesfull! Seconds passed:", time.perf_counter())
    new_Y = pd.Series([0] * N + [1] * count)  # expect count = N, but it is possible that it differs by like 1 or 2

    return new_X, new_Y


def run():
    # Step 1: load and prep the data
    X_frame, Y_frame = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_frame, Y_frame, test_size=0.3, random_state=1234,
                                                        shuffle=True)

    # Step 2: compute frequency seller and add to frame, note test is not included in calculation:
    X_train, X_test = frequency_seller(X_train, X_test)
    X_frame = X_train.reset_index()
    Y_frame = Y_train.reset_index()
    X_known = X_frame[Y_frame != 'Unknown']
    Y_known = Y_frame[Y_frame != 'Unknown']

    X_known.to_csv("data/train_test_frames/final_train_x_onTimeDelivery.csv")
    Y_known.to_csv("data/train_test_frames/final_train_y_onTimeDelivery.csv")

    # * write final train and test set to csv
    X_frame.to_csv("data/train_test_frames/final_train_x.csv")
    Y_frame.to_csv("data/train_test_frames/final_train_y.csv")

    X_test.to_csv("data/train_test_frames/final_test_x.csv")
    Y_test.to_csv("data/train_test_frames/final_test_y.csv")

    return

    # Step 3: obtain train and validation sets by performing 5-fold cv
    train_indices, val_indices = five_fold_cv(X_frame)

    for i in range(0, len(train_indices)):
        X_train, X_val = X_frame.loc[train_indices[i], :], X_frame.loc[val_indices[i], :]

        Y_train, Y_val = Y_frame.loc[train_indices[i], :], Y_frame.loc[val_indices[i], :]

        X_train = X_train.drop(columns={'sellerId', 'orderDate'})
        X_val = X_val.drop(columns={'sellerId', 'orderDate'})

        # Step 4: standardize the data (dummy columns should not be standardized)
        standardized_data_X = standardize_data(X_train)

        # Step 5: perform k-means, for each criteria
        for criteria in ["noCancellation", "Unknown", "onTimeDelivery", "noReturn", "noCase"]:
            if criteria == "Unknown":
                Y_train = Y_train.replace(to_replace="0", value=np.float(1))
                Y_train = Y_train.replace(to_replace=0, value=np.float(1))
                Y_train = Y_train.replace(to_replace="1", value=np.float(1))
                Y_train = Y_train.replace(to_replace=1, value=np.float(1))
                Y_train = Y_train.replace(to_replace="Unknown", value=np.float(0))

                new_train_x, new_train_y = k_means_plus_two_strategies(standardized_data_X, Y_train, "onTimeDelivery",
                                                                       X_train)
            else:
                new_train_x, new_train_y = k_means_plus_two_strategies(standardized_data_X, Y_train, criteria, X_train)

            new_train_x.to_csv("data/train_test_frames/train_x_fold_{0}_{1}.csv".format(i + 1, criteria))
            new_train_y.to_csv("data/train_test_frames/train_y_fold_{0}_{1}.csv".format(i + 1, criteria))

        X_val.to_csv("data/train_test_frames/val_x_fold_{0}.csv".format(i + 1))
        Y_val.to_csv("data/train_test_frames/val_y_fold_{0}.csv".format(i + 1))

        unknown_y = Y_val[Y_val["onTimeDelivery"] != "Unknown"]
        unknown_indices = list(unknown_y.index.values)
        val_x_new = X_val.loc[unknown_indices, :].reset_index()
        val_y_new = Y_val.loc[unknown_indices, :].reset_index()

        val_x_new.tocsv("data/train_test_frames/val_x_fold_{0}_{1}.csv".format(i + 1, "onTimeDelivery"))
        val_y_new.tocsv("data/train_test_frames/val_y_fold_{0}_{1}.csv".format(i + 1, "onTimeDelivery"))

        # Step 6: Generate the full balanced training set for each criteria
        if i == 0:
            stand_data_X_val = standardize_data(X_val)
            for criteria in ["Unknown", "onTimeDelivery", "noCancellation", "noReturn", "noCase"]:
                if criteria == "Unknown":
                    train_x = pd.read_csv("data/train_test_frames/train_x_fold_1_{0}.csv".format(criteria))
                    train_y = pd.read_csv("data/train_test_frames/train_y_fold_1_{0}.csv".format(criteria))
                    print(train_y["0"].value_counts(ascending=True))

                    Y_val = Y_val.replace(to_replace="0", value=np.float(1))
                    Y_val = Y_val.replace(to_replace=0, value=np.float(1))
                    Y_val = Y_val.replace(to_replace="1", value=np.float(1))
                    Y_val = Y_val.replace(to_replace=1, value=np.float(1))
                    Y_val = Y_val.replace(to_replace="Unknown", value=np.float(0))

                    print(Y_train["onTimeDelivery"].value_counts(ascending=True))
                    print(Y_val["onTimeDelivery"].value_counts(ascending=True))

                    new_val_x, new_val_y = k_means_plus_two_strategies(stand_data_X_val, Y_val, "onTimeDelivery", X_val)

                    new_val_x.to_csv("data/train_test_frames/balanced_val_x_fold_1_{0}.csv".format(criteria))
                    new_val_y.to_csv("data/train_test_frames/balanced_val_y_fold_1_{0}.csv".format(criteria))

                    train_x_1 = train_x.append(new_val_x, ignore_index=True)
                    train_x_1.to_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))

                    train_y_1 = train_y["0"].append(new_val_y, ignore_index=True)
                    train_y_1.to_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))

                    return

                if criteria == "onTimeDelivery":
                    # on time delivery prediction consists of two parts unknown/known and known-> true/false
                    # now combine the balanced validation of the fold with the balanced training fold
                    train_x = pd.read_csv("data/train_test_frames/train_x_fold_1_{0}.csv".format(criteria))
                    train_y = pd.read_csv("data/train_test_frames/train_y_fold_1_{0}.csv".format(criteria))
                    train_y = train_y[train_y["0"] != "Unknown"]
                    known_indices = list(train_y.index.values)
                    known_x = train_x.loc[known_indices, :]

                    new_val_x, new_val_y = k_means_plus_two_strategies(stand_data_X_val, Y_val, criteria, X_val)

                    new_val_x.to_csv("data/train_test_frames/balanced_val_x_fold_1_{0}.csv".format(criteria))
                    new_val_y.to_csv("data/train_test_frames/balanced_val_y_fold_1_{0}.csv".format(criteria))

                    train_x_1 = known_x.append(new_val_x, ignore_index=True)
                    train_x_1.to_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))

                    train_y_1 = train_y["0"].append(new_val_y, ignore_index=True)
                    train_y_1.to_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))

                    continue

                new_val_x, new_val_y = k_means_plus_two_strategies(stand_data_X_val, Y_val, criteria, X_val)

                new_val_x.to_csv("data/train_test_frames/balanced_val_x_fold_1_{0}.csv".format(criteria))
                new_val_y.to_csv("data/train_test_frames/balanced_val_y_fold_1_{0}.csv".format(criteria))

                # now combine the balanced test(=validation) of the fold with the balanced training fold
                train_x = pd.read_csv("data/train_test_frames/train_x_fold_1_{0}.csv".format(criteria))
                train_y = pd.read_csv("data/train_test_frames/train_y_fold_1_{0}.csv".format(criteria))["0"]

                train_x_1 = train_x.append(new_val_x, ignore_index=True)
                train_x_1.to_csv("data/train_test_frames/balanced_train_x_{0}.csv".format(criteria))

                train_y_1 = train_y.append(new_val_y, ignore_index=True)
                train_y_1.to_csv("data/train_test_frames/balanced_train_y_{0}.csv".format(criteria))

            return


run()
