import pandas as pd

file_path = "data/data_2020.csv"
frame = pd.read_csv(file_path)
frame2 = frame.to_numpy()

print(frame.keys())
print(frame2[0])
