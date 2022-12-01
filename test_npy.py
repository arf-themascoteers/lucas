import numpy as np
from functools import partial

def si(mat, x, y):
    return (mat[:,x] - mat[:,y]) / (mat[:,x] + mat[:,y])

def gen_si(x, y):
    return partial(si, x = x, y = y)

x = np.array([
    [1,2,3,4,5],
    [1,2,3,4,5],
    [1,2,3,4,5]
])

si_fun = gen_si(1,3)
sim = si_fun(x)
print(sim)