# load modules and packages
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import logging
import time
from numba import jit
import faiss


def load_data():
    """ This function loads and prepares the data """
    frame = pd.read_csv("data/frame.csv",
                        dtype={'onTimeDelivery': str, 'datetTimeFirstDeliveryMoment': object, 'returnCode': object,
                               'transporterNameOther': object,
                               'cancellationReasonCode': object})

    frame_X = frame[['sellerId', 'orderDate', 'countryCode', 'fulfilmentType', 'productGroup', 'countryOriginSeller',
                     'currentCountryAvailabilitySeller', 'totalPrice', 'quantityOrdered', 'promisedDeliveryDate',
                     'registrationDateSeller', 'day_of_week', 'month_of_year']]

    X = frame_X.head(1000000)
    X = frame_X

    categorical_variables = ['countryCode', 'fulfilmentType', 'productGroup', 'countryOriginSeller',
                             'currentCountryAvailabilitySeller', 'day_of_week', 'month_of_year']
    X_frame = pd.get_dummies(X, columns=categorical_variables)

    frame_Y = frame[['noCancellation', 'onTimeDelivery', 'noReturn', 'noCase']]

    Y = frame_Y.head(1000000)
    Y = frame_Y

    Y = Y.replace(to_replace=True, value=np.float(1))
    Y = Y.replace(to_replace=False, value=np.float(0))
    Y = Y.replace(to_replace="true", value=np.float(1))
    Y_frame = Y.replace(to_replace="false", value=np.float(0))
    return X_frame, Y_frame


def five_fold_cv(X):
    """ This function splits the input data in 5 training sets with corresponding test sets """
    np.random.seed(1234)
    kf = KFold(n_splits=5, shuffle=True)

    train_indices = []
    test_indices = []
    for train_index, test_index in kf.split(X):
        train_indices.append(train_index)
        test_indices.append(test_index)

    with open('data/train_indices.pkl', 'wb') as f:
        pickle.dump(train_indices, f)
    f.close()

    with open('data/test_indices.pkl', 'wb') as f:
        pickle.dump(test_indices, f)
    f.close()
    return train_indices, test_indices


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

    print(len(majority_rows))
    print(len(minority_rows))

    # step 5.2: perform kmeans clustering
    n = len(minority_class)  # = number of minority class variables
    if n == 0:
        logging.warning("There are no observations in the minority class!")
        empty_frame = pd.DataFrame([])
        empty_series = pd.Series([])
        return empty_frame, empty_frame, empty_series

    print(time.perf_counter())
    # n = 5 # EVEN HANDMATIG OP 10 GEZET, ANDERS te lange running time ivm testen
    majority_data_stdz = standardized_data_x.loc[majority_rows, :]
    majority_data = normal_data_x.loc[majority_rows, :]
    kmeans = MiniBatchKMeans(n_clusters=n, batch_size=n).fit(majority_data_stdz)
    print(time.perf_counter())
    majority_data['cluster'] = kmeans.predict(majority_data_stdz)
    print(majority_data)
    print(time.perf_counter())

    # step 5.2.2: print some count information about the clusters
    # print('Summary:')
    summary = majority_data.groupby(['cluster']).mean()
    # summary['count'] = majority_data['cluster'].value_counts()
    # summary = summary.sort_values(by='count', ascending=False)
    print(summary)
    print(time.perf_counter())
    # step 5.3: find the values for first strategy
    minority_data = normal_data_x.loc[minority_rows, :]
    strategy_1_X = minority_data.append(majority_data.groupby(['cluster']).mean(), ignore_index=True)
    print(time.perf_counter())

    # step 5.4: find the values for the second strategy
    # compute nearest neighbor of a given cluster within that cluster instead of entire sample
    def nearest_neighbor(data, center, cluster_number):
        """ This function finds the nearest neighbor given a frame and a cluster center"""
        data = data[data["cluster"] == cluster_number]
        data = data.drop(columns={'cluster'})
        # print(data.keys() == center.keys()) columns should be the same and respective positions identical

        distances = distance.cdist(pd.DataFrame.to_numpy(data), [np.array(center)], metric='euclidean')
        result = np.where(distances == np.amin(distances))[0]
        nn = data.iloc[result, :]
        return nn

    print(time.perf_counter())
    strategy_2_X = minority_data
    for c in range(0, n):
        n_c = nearest_neighbor(majority_data, summary.loc[c, :], c)
        strategy_2_X = strategy_2_X.append(n_c, ignore_index=True)

    both_strategies_Y = pd.Series([0] * n + [1] * n)
    print(time.perf_counter())
    return strategy_1_X, strategy_2_X, both_strategies_Y


def run():
    # Step 1: load and prep the data
    X_frame, Y_frame = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X_frame, Y_frame, test_size=0.3, random_state=1234,
                                                        shuffle=True)

    # Step 2: compute frequency seller and add to frame, note test is not included in calculation:
    X_train, X_test = frequency_seller(X_train, X_test)
    X_frame = X_train.reset_index()
    Y_frame = Y_train.reset_index()

    # * write final test set to csv
    file_name_final_test_x = "data/train_test_frames/final_test_x.csv"
    X_test.to_csv(file_name_final_test_x)
    file_name_final_test_y = "data/train_test_frames/final_test_y.csv"
    Y_test.to_csv(file_name_final_test_y)

    # Step 3: obtain train and test sets by performing 5-fold cv
    train_indices, test_indices = five_fold_cv(X_frame)

    for i in range(0, len(train_indices)):
        X_train, X_test = X_frame.loc[train_indices[i], :], X_frame.loc[test_indices[i], :]

        Y_train, Y_test = Y_frame.loc[train_indices[i], :], Y_frame.loc[test_indices[i], :]

        X_train = X_train.drop(columns={'sellerId', 'orderDate'})
        X_test = X_test.drop(columns={'sellerId', 'orderDate'})

        # Step 4: standardize the data (dummy columns should not be standardized)
        standardized_data_X = standardize_data(X_train)

        # Step 5: perform k-means, for each criteria
        for criteria in ["noCancellation", "noReturn", "noCase"]:
            train_x_1, train_x_2, train_y = k_means_plus_two_strategies(standardized_data_X, Y_train, criteria, X_train)

            file_name_1 = "data/train_test_frames/train_x_str1_fold_{0}_{1}.csv".format(i + 1, criteria)
            train_x_1.to_csv(file_name_1)

            file_name_2 = "data/train_test_frames/train_x_str2_fold_{0}_{1}.csv".format(i + 1, criteria)
            train_x_2.to_csv(file_name_2)

            file_name_3 = "data/train_test_frames/train_y_fold_{0}_{1}.csv".format(i + 1, criteria)
            train_y.to_csv(file_name_3)

        file_name_4 = "data/train_test_frames/test_x_fold_{0}.csv".format(i + 1)
        X_test.to_csv(file_name_4)

        file_name_5 = "data/train_test_frames/test_y_fold_{0}.csv".format(i + 1, )
        Y_test.to_csv(file_name_5)

        # the criterion onTimeDelivery there is a third variable (unknown) this class is predicted before using any
        # of the algorithms, hence the KMeans strategies are not needed for the unknown class, only for the boolean
        # cases
        train_x_1, train_x_2, train_y = k_means_plus_two_strategies(standardized_data_X, Y_train, "onTimeDelivery",
                                                                    X_train)
        unknown_y = Y_train[Y_train["onTimeDelivery"] == "Unknown"]
        unknown_indices = list(unknown_y.index.values)
        unknown_observations = X_train.loc[unknown_indices, :]

        train_y = pd.Series(["Unknown"] * len(unknown_y) + list(train_y))
        train_x_1 = unknown_observations.append(train_x_1, ignore_index=True)
        train_x_2 = unknown_observations.append(train_x_2, ignore_index=True)

        file_name_1 = "data/train_test_frames/train_x_str1_fold_{0}_{1}.csv".format(i + 1, "onTimeDelivery")
        train_x_1.to_csv(file_name_1)

        file_name_2 = "data/train_test_frames/train_x_str2_fold_{0}_{1}.csv".format(i + 1, "onTimeDelivery")
        train_x_2.to_csv(file_name_2)

        file_name_3 = "data/train_test_frames/train_y_fold_{0}_{1}.csv".format(i + 1, "onTimeDelivery")
        train_y.to_csv(file_name_3)


run()
