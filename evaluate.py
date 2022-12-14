import torch
from train import train
from test import test
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def calculate_r2(train_ds, test_ds, model):
    model_instance = None
    if model == "nn":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_instance = train(device, train_ds)
        return test(device, test_ds, model_instance)
    else:
        train_x = train_ds.get_x()
        train_y = train_ds.get_y()
        test_x = test_ds.get_x()
        test_y = test_ds.get_y()

        if model == "linear":
            model_instance = LinearRegression()
        elif model == "rf":
            model_instance = RandomForestRegressor(max_depth=5, n_estimators=1000)

        model_instance = model_instance.fit(train_x, train_y)
        return model_instance.score(test_x, test_y)


def r2_once(dm, model):
    r2 = calculate_r2(dm.get_train_ds(), dm.get_test_ds(), model)
    print(r2)
    return r2


def r2(dm, model):
    r2s = []
    for train_ds, test_ds in dm.get_10_folds():
        r2 = calculate_r2(train_ds, test_ds, model)
        print(r2)
        r2s.append(r2)

    return np.array(r2s)