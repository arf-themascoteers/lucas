import ds_manager
import numpy as np
import linear
from functools import partial

MIN_WAVELENGTH = 400
MAX_WAVELENGTH = 2350
DIFF = 50
size = ((MAX_WAVELENGTH - MIN_WAVELENGTH) // DIFF) + 1
X_ARRAY_DIFF = 0.5
X_ARRAY_INDEX_JUMP = DIFF // X_ARRAY_DIFF

matrix = np.zeros([size, size])

np.save("nps/matrix.npy", matrix)

dm = ds_manager.DSManager()
train_ds = dm.get_train_ds()
test_ds = dm.get_test_ds()

train_y = train_ds.get_si(None)
test_y = test_ds.get_si(None)


def si(mat, x, y):
    return (mat[:,x] - mat[:,y]) / (mat[:,x] + mat[:,y])


def gen_si(x, y):
    return partial(si, x = x, y = y)


for i in matrix.shape[0]:
    for j in matrix.shape[1]:
        si_fun = gen_si(i, j)
        train_x = train_ds.get_si(None)
        test_x = test_ds.get_si(None)
        matrix[i][j] = linear.get_r2(train_x, train_y, test_x, test_y)

    print(f"Done {i} among {size}")
    np.save("nps/matrix.npy", matrix)

print("done")