"""
This script contains functions that can be used to analyze the dataset
(These are not used in a main file)
"""
# Packages and modules
import time
import pandas as pd
import numpy as np
from collections import Counter

from networkx.drawing.tests.test_pylab import plt

# Import the data
frame = pd.read_csv("data/frame.csv",
                    dtype={'onTimeDelivery': str, 'datetTimeFirstDeliveryMoment': object, 'returnCode': object,
                           'transporterNameOther': object, 'cancellationReasonCode': object})


# Function that finds characteristics of each column in the given frame
def find_range(f):
    for column in f.keys():
        col = f[column]
        print("Characteristic name: " + column)  # prints out which column is being analyzed
        print("Total column length: " + str(len(col)))

        col.dropna(inplace=True)  # Remove nan's/ None's

        print("Number of items(None removed): " + str(
            len(col)))  # prints out how many actual observations the column has
        print("Number of unique items in column: " + str(col.nunique()))
        if col.nunique() < 50:
            print("Set of items: " + str(col.unique()))
        try:
            col = np.array(col)
            col_without_unknown = col[col != "Unknown"]  # Remove the Unknown observations if present
            col_without_unknown = col_without_unknown.astype("float64")
            print("Minimum value (None removed): " + str(np.min(col_without_unknown)))
            print("Maximum value (None removed): " + str(np.max(col_without_unknown)))
            try:
                print("Median value: " + str(np.median(col_without_unknown)))
            except:
                pass
        except:
            pass
        print("\n")


# Function that counts the occurrences of all values in a given column
def count_classes(column_name, f):
    col = f[column_name]
    return col.value_counts(ascending=True)


# This function groups a frame by specified columns
def groupby(frame, column1, column2):
    counts = frame.groupby([column1, column2]).size()
    return counts


# Function that counts classes belonging to a certain column given that the order has not met a criterion,
# also computes ratio
def most_criteria(f, col_name, criterion):
    not_met = f[[col_name, criterion]]

    met = not_met[not_met[criterion] is True]
    not_met = not_met[not_met[criterion] is False]

    not_met_sorted = count_classes(col_name, not_met)
    met_sorted = count_classes(col_name, met)

    relative = not_met_sorted.divide(met_sorted, fill_value=None).dropna().sort_values()

    return not_met_sorted, relative


# Function that counts classes belonging to a certain column given that the order is not delivered on time,
# also calculates ratio to no on time deliveries
def least_on_time_delivery(f, col_name):
    late = f[[col_name, 'onTimeDelivery']]
    on_time = late[late['onTimeDelivery'] == "True"]
    cases = late[late['onTimeDelivery'] == "False"]

    late_sorted = count_classes(col_name, cases)
    on_time_sorted = count_classes(col_name, on_time)
    relative_late_deliveries = late_sorted.divide(on_time_sorted, fill_value=None).dropna().sort_values()
    return late_sorted, relative_late_deliveries


# Function that calculates the class ratios
def class_ratios(f):
    total_orders = len(f)

    return_amount = len(f[f["noReturn"] is False])
    return_ratio = np.divide(total_orders, return_amount)

    cancellation_amount = len(f[f["noCancellation"] is False])
    cancellation_ratio = np.divide(total_orders, cancellation_amount)

    # Note that this ratio is late to ontime+unknowns
    late_delivery_amount = len(f[f["onTimeDelivery"] == "false"])
    late_delivery_ratio = np.divide(total_orders, late_delivery_amount)

    case_amount = len(f[f["noCase"] is False])
    case_ratio = np.divide(total_orders, case_amount)

    return return_ratio, cancellation_ratio, late_delivery_ratio, case_ratio


# column can choose from cancellationDate [11], datetTimeFirstDeliveryMoment
# (promisedDeliveryDate) [31], startDateCase [31], returnDateTime [31]
def column_and_order_dates(data, column):
    columndates = data[column]
    columndates = columndates.dropna()
    indices = np.array(columndates.index)
    orderdates = data.loc[indices, "orderDate"]

    columndates = pd.to_datetime(columndates).dt.date
    orderdates = pd.to_datetime(orderdates).dt.date

    return columndates, orderdates


# This function transforms a given column containing dates into a days difference column
def days_difference(columndates, orderdates, column_days):
    difference_dates = (columndates - orderdates) / np.timedelta64(1, 'D')
    difference_dates = difference_dates[difference_dates > -1]
    amount = len(difference_dates)
    difference_dates = difference_dates[difference_dates < column_days]

    date_values = difference_dates.value_counts().sort_index()
    date_percentages = date_values / amount
    dates = np.transpose([date_values, date_percentages])
    date_values = pd.DataFrame(dates, columns=["Sum per day", "Percentages"])
    return difference_dates, date_values


# This function computes the Pearson correlation of the criteria
def correlations_categorical(data):
    main_variables = data[["noCancellation", "onTimeDelivery", "noCase", "noReturn"]]
    factorized_vars = main_variables.apply(lambda x: pd.factorize(x)[0])
    correlation = factorized_vars.corr(method="pearson")
    return correlation


# This function plots a histogram given a column
def get_hist(frame, col):
    [columndates, orderdates] = column_and_order_dates(frame, col)
    [difference_dates, date_values] = days_difference(columndates, orderdates, 31)

    plt.hist(x=difference_dates, bins=100)
    plt.xlabel('Day')
    plt.ylabel('Amount')
    plt.show()
    print(date_values)


# With this function we can compute how many orders are caused by not meeting one or more criteria
# Note that we utilized this function by manually changing criteria and their status
def matches_causes_count(frame):
    print(frame[frame["detailedMatchClassification"] == "KNOWN HAPPY"])
    print(frame["returnDateTime"].unique())
    print(frame[(frame['detailedMatchClassification'] == "UNKNOWN") & (frame['noReturn'] == True) & (
            frame['noCase'] == True) & (frame['onTimeDelivery'] == 'Unknown') & (frame['noCancellation'] == True)])
    print("*" * 25)
    print(frame[(frame['detailedMatchClassification'] == "UNHAPPY") & (frame['noReturn'] == True) & (
            frame['noCase'] == True) & (frame['onTimeDelivery'] == 'Unknown') & (frame['noCancellation'] == True)])
    print(frame[["noReturn", "returnDateTime"]])
    print(frame['detailedMatchClassification'].unique())
    print(frame[(frame['returnDateTime'].astype(float) < 30)])
    print(frame[(frame['returnDateTime'].astype(float) > 30) & (frame['onTimeDelivery'] == 'True') & (
            frame['noCase'] == True) & (frame['noCancellation'] == True)])

    print("*" * 25)


