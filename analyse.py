import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

MIN_WAVELENGTH = 400
MAX_WAVELENGTH = 2399
DIFF = 1
size = ((MAX_WAVELENGTH - MIN_WAVELENGTH) // DIFF) + 1
X_ARRAY_DIFF = 0.5

def transform_index(ind):
    return ind * DIFF + MIN_WAVELENGTH

def get_bands(index):
    unraveled = np.unravel_index(index, (size, size))
    return transform_index(unraveled[0]), transform_index(unraveled[1])

def anal():
    array = np.load("nps/ndi4.npy")
    array = array.flatten()
    indices = np.argsort(array)
    for i in range(100):
        index = indices[len(indices) - 1 - i]
        b1, b2 = get_bands(index)
        print(b1, b2, array[index])
    print(max(array))

# array = np.load("nps/matrix2.npy")
# print(transform_index(array.shape[0]-1))

anal()