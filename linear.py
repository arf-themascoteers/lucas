import numpy

import lucas_dataset
from sklearn.linear_model import LinearRegression
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
    reg = LinearRegression().fit(x,y)

    print("Train done")
    end = time.time()
    required = end - start
    print(f"Train seconds: {required}")

    pickle.dump(reg, open("models/linear3","wb"))

def test():
    reg = pickle.load(open('models/linear3', 'rb'))
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
    print("R2",r2_score(y, y_hat))
    print("MSE",mean_squared_error(y, y_hat))

    # for i in range(10):
    #     a_y = y[i]
    #     a_y_hat = y_hat[i]
    #     print(f"{a_y:.3f}\t\t{a_y_hat:.3f}")

def dump():
    reg = pickle.load(open('models/linear2', 'rb'))
    x = abs(reg.coef_)
    x = numpy.array(x)
    y = numpy.argsort(x)
    for i in range(len(y)):
        z = y[i]
        # print(z,(z * 0.5)+400 , x[z])
        a = (z * 0.5) + 400
        b = z+11
        print(f"{b},",end="")
    #
train()
test()
#dump()