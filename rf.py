import lucas_dataset
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


def get_r2(train_x, train_y, test_x, test_y):
    reg = RandomForestRegressor(max_depth=15, n_estimators=100).fit(x,y)
    return reg.score(test_x, test_y)

