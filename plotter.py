import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import numpy
import pandas as pd
import lucas_dataset

ds = lucas_dataset.LucasDataset(is_train=True)
x = ds.get_x()
y = ds.get_y()
for i in range(10):
    plt.plot(x[i])
    plt.show()


print("done")