import lucas_dataset
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


def train():
    ds = lucas_dataset.LucasDataset(is_train=True)
    x = ds.get_x()
    y = ds.get_y()
    aux = ds.get_aux()
    new_x = np.concatenate((x,aux), axis=1)
    start = time.time()
    reg = RandomForestRegressor(max_depth=15, n_estimators=1000).fit(x,y)

    print("Train done")
    end = time.time()
    required = end - start
    print(f"Train seconds: {required}")
    print(f"Depth: {reg.tree_.max_depth}")

    pickle.dump(reg, open("models/rf","wb"))


def test():
    reg = pickle.load(open('models/rf', 'rb'))
    ds = lucas_dataset.LucasDataset(is_train=False)
    x = ds.get_x()
    y = ds.get_y()
    aux = ds.get_aux()
    new_x = np.concatenate((x,aux), axis=1)
    start = time.time()
    y_hat = reg.predict(x)
    end = time.time()
    required = end - start
    print(f"Test seconds: {required}")
    print(r2_score(y, y_hat))
    print(mean_squared_error(y, y_hat))

    for i in range(10):
        a_y = ds.unscale(y[i])
        a_y_hat = ds.unscale(y_hat[i])
        print(f"{a_y:.3f}\t\t{a_y_hat:.3f}")

train()
test()