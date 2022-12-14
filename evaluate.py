import torch
from train import train
from test import test
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def r2(dm, model):
    r2s = []
    for train_ds, test_ds in dm.get_10_folds():
        r2 = 0
        model_instance =None
        if model == "nn":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_instance = train(device, train_ds)
            r2 = test(device, test_ds, model_instance)
        else:
            train_x = train_ds.get_x()
            train_y = train_ds.get_y()
            test_x = test_ds.get_x()
            test_y = test_ds.get_y()

            if model == "linear":
                model_instance = LinearRegression()
            elif model == "rf":
                model_instance = RandomForestRegressor(max_depth=2, n_estimators=100)

            model_instance = model_instance.fit(train_x,train_y)
            r2 = model_instance.score(test_x, test_y)

        r2s.append(r2)

    return np.array(r2s)