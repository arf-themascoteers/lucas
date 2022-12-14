import numpy as np
import pandas as pd
import os

cols = ["col1","col2"]
rows = ["row1","row2", "row3", "row4"]

data = np.zeros((len(rows), len(cols)))
path = "testfile.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    df.drop(columns=df.columns[0], axis=1, inplace=True)
    part_data = df.to_numpy()
    data[0:part_data.shape[0],0:part_data.shape[1]] = part_data


for index_row, row in enumerate(rows):
    for index_col, col in enumerate(cols):
        if data[index_row][index_col]  != 0:
            print("Pre", data[index_row][index_col])
        else:
            val = (index_row+1) * (index_col+2)
            print("evaluated", val)
            data[index_row][index_col] = val
            df = pd.DataFrame(data=data, columns=cols, index=rows)
            df.to_csv(path)

