import pandas as pd
import os
from sklearn import model_selection

csv_file_location = "data/min.csv"
df = pd.read_csv(csv_file_location)
train, test = model_selection.train_test_split(df, test_size=0.1)
test.to_csv("data/miner.csv", index=False)
print("done")
