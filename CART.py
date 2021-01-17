import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from io import StringIO
from IPython.display import Image
import pydotplus

# Importing data
frame_2019 = pd.read_csv("data/data_2019.csv")  # 2110338
frame_2020 = pd.read_csv("data/data_2020.csv")  # 2645037
frame = pd.concat([frame_2019, frame_2020], ignore_index=True)  # 4755375

# Splitting data
X = frame[['totalPrice', 'quantityOrdered', 'cntDistinctCaseIds']]
Y = frame['noCancellation']
X[np.isnan(X)] = 0
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1234)

# CART
model = DecisionTreeRegressor(max_leaf_nodes=20)
# one_hot_data = pd.get_dummies(frame[['A', 'B', 'C']], drop_first=True)
rt = model.fit(X_train, Y_train)
dot_data = StringIO()
export_graphviz(rt, out_file=dot_data)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


