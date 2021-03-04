import numpy as np
import pandas as pd
import DataImbalance

# Import the data
missing_value_formats = ["n.a.", "?", "NA", "n/a", "na", "--", "NaN", " ", ""]
frame_2019 = pd.read_csv("data/data_2019.csv", na_values=missing_value_formats,
                         dtype={'onTimeDelivery': str, 'datetTimeFirstDeliveryMoment': object, 'returnCode': object,
                                'transporterNameOther': object, 'cancellationReasonCode': object})
frame_2020 = pd.read_csv("data/data_2020.csv", na_values=missing_value_formats,
                         dtype={'onTimeDelivery': str, 'datetTimeFirstDeliveryMoment': object, 'returnCode': object,
                                'transporterNameOther': object, 'cancellationReasonCode': object})
frame = pd.concat([frame_2019, frame_2020], ignore_index=True)


# This function creates a new column containing the day of the week orders were placed
def day_of_the_week(f):
    f['orderDate'] = pd.to_datetime(f['orderDate'])
    f['day_of_week'] = f['orderDate'].dt.day_name()
    return f


# This function creates a new column containing the month of the year the orders were placed
def month_of_year(f):
    f['orderDate'] = pd.to_datetime(f['orderDate'])
    f['month_of_year'] = f['orderDate'].dt.month_name()
    return f


# Function to determine the days difference between two dates
def days_difference(data, column, origin):
    columndates = data[column]
    columndates = columndates.dropna()
    indices = np.array(columndates.index)
    orderdates = data.loc[indices, origin]

    columndates = pd.to_datetime(columndates).dt.date
    orderdates = pd.to_datetime(orderdates).dt.date

    difference_dates = (columndates - orderdates) / np.timedelta64(1, 'D')
    difference_dates = difference_dates[difference_dates > -1]

    return difference_dates


# Determine the days difference between the order date, as it gives more information
frame["cancellationDate"] = days_difference(frame, "cancellationDate", "orderDate")
frame["promisedDeliveryDate"] = days_difference(frame, "promisedDeliveryDate", "orderDate")
frame["shipmentDate"] = days_difference(frame, "shipmentDate", "orderDate")
frame["datetTimeFirstDeliveryMoment"] = days_difference(frame, "datetTimeFirstDeliveryMoment", "orderDate")
frame["startDateCase"] = days_difference(frame, "startDateCase", "orderDate")
frame["returnDateTime"] = days_difference(frame, "returnDateTime", "orderDate")
frame["registrationDateSeller"] = days_difference(frame, "orderDate", "registrationDateSeller")

# Drop the orders that give NA values for these observations
frame = frame.dropna(subset=['registrationDateSeller'])
frame = frame.dropna(subset=['promisedDeliveryDate'])

# Dropping the orders that should not have a NA value
frame = frame.drop(frame[(frame.transporterCode == 'OTHER') & (pd.isna(frame.transporterNameOther))].index)
frame = frame.drop(frame[(frame.transporterCode != 'OTHER') & (pd.notna(frame.transporterNameOther))].index)

# Transforming missing values (NAs) into Unknown or drop them
# Returns
frame = frame.drop(frame[(frame.noReturn == False) & (pd.isna(frame.returnDateTime))].index)
frame['quanityReturned'].fillna("Unknown", inplace=True) # when a product is not returned, this still gives unknown instead of NA ()
frame['returnCode'].fillna("Unknown", inplace=True)

# Delivery
frame = frame.drop(frame[(frame.onTimeDelivery == "True") & (pd.isna(frame.datetTimeFirstDeliveryMoment))].index)
frame['onTimeDelivery'].fillna("Unknown", inplace=True)
frame['datetTimeFirstDeliveryMoment'].fillna("Unknown", inplace=True)
frame['transporterCode'].fillna("Unknown", inplace=True)
frame['transporterName'].fillna("Unknown", inplace=True)

# Cases
frame = frame.drop(frame[(frame.noCase == False) & (pd.isna(frame.startDateCase))].index)
frame['cntDistinctCaseIds'].fillna("Unknown", inplace=True)

# Cancellations
frame = frame.drop(frame[(frame.noCancellation == False) & (pd.isna(frame.cancellationDate))].index)
frame['cancellationReasonCode'].fillna("Unknown", inplace=True)

# Adding new variables
frame = day_of_the_week(frame)
frame = month_of_year(frame)

# Saving the transformed frame
frame.to_csv('frame.csv')

# Create all the folds and make balanced subsets balanced where needed
# WARNING: takes a lot of time
DataImbalance.run()
