import ds_manager
import numpy as np
import linear
from functools import partial

MIN_WAVELENGTH = 400
MAX_WAVELENGTH = 2399
DIFF = 1
size = ((MAX_WAVELENGTH - MIN_WAVELENGTH) // DIFF) + 1
X_ARRAY_DIFF = 0.5

def transform_index(index):
    wavelenth = MIN_WAVELENGTH + (index * DIFF)
    new_index = (wavelenth - MIN_WAVELENGTH) / X_ARRAY_DIFF
    return int(new_index)


def si(mat, x, y):
    x = transform_index(x)
    y = transform_index(y)
    return ((mat[:, x] - mat[:, y]) / (mat[:, x] + mat[:, y])).reshape(-1, 1)

def run_plz():
    matrix = np.zeros([size, size])

    np.save("nps/matrix.npy", matrix)

    dm = ds_manager.DSManager()
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()

    train_y = train_ds.get_y()
    test_y = test_ds.get_y()

    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            si_fun = partial(si, x=i, y=j)
            train_x = train_ds.get_si(si_fun)
            test_x = test_ds.get_si(si_fun)
            matrix[i][j] = linear.get_r2(train_x, train_y, test_x, test_y)

        print(f"Done {i} among {size}")
        np.save("nps/matrix.npy", matrix)

    print("done")

