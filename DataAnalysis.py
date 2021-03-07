import time
import pandas as pd
import numpy as np
from collections import Counter

from networkx.drawing.tests.test_pylab import plt

frame = pd.read_csv("data/frame.csv",
                    dtype={'onTimeDelivery': str, 'datetTimeFirstDeliveryMoment': object, 'returnCode': object,
                           'transporterNameOther': object,
                           'cancellationReasonCode': object})  # 4778046  4778215 (-1420 compared to original data)


# Below function finds characteristics of each column in the given frame
def find_range(f):
    for column in f.keys():
        col = f[column]
        print("Characteristic name: " + column)
        print("Total column length: " + str(len(col)))

        col.dropna(inplace=True)

        print("Number of items(None removed): " + str(len(col)))
        print("Number of unique items in column: " + str(col.nunique()))
        if col.nunique() < 50:
            print("Set of items: " + str(col.unique()))
        try:
            col = np.array(col)
            # col_without_unknown = np.delete(col, np.where(col == "Unknown"))
            col_without_unknown = col[col != "Unknown"]
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


# Below function is the slow version of the find_range function
def check_unique_values(f):
    unique_to_dict = {}
    for column in f.keys():
        if column == 'cntDistinctCaseIds' or column == 'transporterNameOther':
            continue
        temp_set = set(f[column])
        unique_to_dict[column] = temp_set
        print("Characteristic name: " + column)
        print("Number of unique items: " + str(len(temp_set)))
        if len(temp_set) < 25:
            print("Items in set: " + str(temp_set))
        try:
            print("Minimum value: " + str(min(temp_set)))
            print("Maximum value: " + str(max(temp_set)))
            try:
                print("Median value: " + str(np.median(np.array(f[column]))))
            except:
                pass
        except:
            find_range(f[column])
        print("\n")
        time.sleep(20)


# Below function counts the occurrences of all values in a given column
def count_classes(column_name, f):
    col = f[column_name]
    return col.value_counts(ascending=True)


# This function counts the classes belong to a certain column given that the order is returned, also ratio to no returns
def most_returns(f, col_name):
    # first find product group with most returns
    returns = f[[col_name, 'noReturn']]
    no_returns = returns[returns['noReturn'] == True]
    returns = returns[returns['noReturn'] == False]
    returns_sorted = count_classes(col_name, returns)
    no_returns_sorted = count_classes(col_name, no_returns)
    relative = returns_sorted.divide(no_returns_sorted, fill_value=None).dropna().sort_values()
    return returns_sorted, relative


# This function counts the classes belong to a certain column given that the order is cancelled, also ratio to no
# cancels
def most_cancellation(f, col_name):
    # first find product group with most cancellations
    cancelled = f[[col_name, 'noCancellation']]
    not_cancelled = cancelled[cancelled['noCancellation'] == True]
    cancelled = cancelled[cancelled['noCancellation'] == False]
    cancels_sorted = count_classes(col_name, cancelled)
    no_cancels_sorted = count_classes(col_name, not_cancelled)
    relative_cancels = cancels_sorted.divide(no_cancels_sorted, fill_value=None).dropna().sort_values()
    return cancels_sorted, relative_cancels


# This function counts the classes belong to a certain column given that the order has a case, also ratio to no cases
def most_cases(f, col_name):
    # first find product group with most cases
    cases = f[[col_name, 'noCase']]
    no_cases = cases[cases['noCase'] == True]
    cases = cases[cases['noCase'] == False]
    cases_sorted = count_classes(col_name, cases)
    no_cases_sorted = count_classes(col_name, no_cases)
    relative_cases = cases_sorted.divide(no_cases_sorted, fill_value=None).dropna().sort_values()
    return cases_sorted, relative_cases


# This function counts the classes belong to a certain column given that the order is not delivered on time,
# also ratio to on time deliveries
def least_on_time_delivery(f, col_name):
    # first find product group with least on time deliveries
    late = f[[col_name, 'onTimeDelivery']]
    on_time = late[late['onTimeDelivery'] == "True"]
    cases = late[late['onTimeDelivery'] == "False"]

    late_sorted = count_classes(col_name, cases)
    on_time_sorted = count_classes(col_name, on_time)
    relative_late_deliveries = late_sorted.divide(on_time_sorted, fill_value=None).dropna().sort_values()
    return late_sorted, relative_late_deliveries


# Below function contains code that might be useful in the future, but can be deleted if data analysis is complete
def storage():
    cancellation = frame[["currentCountryAvailabilitySeller", "sellerId"]]
    cancellation = cancellation.loc[np.where(pd.isnull(cancellation))[0]]
    print(cancellation)
    date = cancellation['sellerId']
    print(date.nunique())
    print(date.unique())


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


def correlations_categorical(data):
    main_variables = data[["noCancellation", "onTimeDelivery", "noCase", "noReturn"]]
    factorized_vars = main_variables.apply(lambda x: pd.factorize(x)[0])
    correlation = factorized_vars.corr(method="pearson")
    return correlation


def groupby(data, column1, column2):
    counts = frame.groupby([column1, column2]).size()
    return counts


def thijs_functie():
    [columndates, orderdates] = column_and_order_dates(frame, "returnDateTime")
    [difference_dates, date_values] = days_difference(columndates, orderdates, 31)

    plt.hist(x=difference_dates, bins=100)
    plt.xlabel('Day')
    plt.ylabel('Amount')
    plt.show()
    print(date_values)

    correlation = correlations_categorial(frame)
    counts = groupby(frame, "noCase", "noCancellation")
    counts1 = groupby(frame, "noCase", "onTimeDelivery")
    counts2 = groupby(frame, "noCase", "noReturn")
    counts3 = groupby(frame, "noCancellation", "onTimeDelivery")
    counts4 = groupby(frame, "noCancellation", "noReturn")
    counts5 = groupby(frame, "onTimeDelivery", "noReturn")


def rest_code(frame):
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
            frame['noCase'] == True) & (frame['noCancellation'] == True)])  # 105076
    # print(frame.loc[807032, :])

    print("*" * 25)


# This function calculates the class ratios
def class_ratios(f):
    total_orders = len(f)

    return_amount = len(f[f["noReturn"] == False])
    return_ratio = np.divide(total_orders, return_amount)

    cancellation_amount = len(f[f["noCancellation"] == False])
    cancellation_ratio = np.divide(total_orders, cancellation_amount)

    late_delivery_amount = len(f[f["onTimeDelivery"] == "false"])
    late_delivery_ratio = np.divide(total_orders, late_delivery_amount)

    case_amount = len(f[f["noCase"] == False])
    case_ratio = np.divide(total_orders, case_amount)

    return return_ratio, cancellation_ratio, late_delivery_ratio, case_ratio


