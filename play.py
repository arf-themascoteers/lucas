import pandas as pd
from sklearn import model_selection

out = "data/lucasmid.csv"
df = pd.read_csv("data/lucas.csv")

train, test = model_selection.train_test_split(df, test_size=0.3)
test.to_csv(out, index=False)
print("Done")