import numpy

import lucas_dataset
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import time


def train(x, y):
    start = time.time()
    reg = LinearRegression().fit(x,y)

    #print("Train done")
    end = time.time()
    required = end - start
    #print(f"Train seconds: {required}")

    #pickle.dump(reg, open("models/linear3","wb"))
    return reg

def test(reg, x, y):
    #reg = pickle.load(open('models/linear3', 'rb'))

    start = time.time()
    y_hat = reg.predict(x)
    end = time.time()
    required = end - start
    r2 = r2_score(y, y_hat)
    #print(f"Test seconds: {required}")
    # print("R2",r2)
    # print("MSE",mean_squared_error(y, y_hat))
    return r2
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


if __name__ == "__main__":
    train_ds = lucas_dataset.LucasDataset(is_train=True)
    test_ds = lucas_dataset.LucasDataset(is_train=False)

    train_x = train_ds.get_x()
    train_y = train_ds.get_y()
    train_new_x = train_ds.get_new_x()

    test_x = test_ds.get_x()
    test_y = test_ds.get_y()
    test_new_x = test_ds.get_new_x()

    reg = train(train_x, train_y)
    without_si = test(reg, test_x, test_y)

    reg = train(train_new_x,train_y)
    with_si = test(reg, test_new_x, test_y)

    print(without_si, with_si)