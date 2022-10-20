import pywt
import lucas_dataset
import matplotlib.pyplot as plt

ds = lucas_dataset.LucasDataset()
x = ds.get_x()
y = ds.get_y()

x1 = x[0]
tups = pywt.wavedec(x1, 'db1', level=3)

plt.plot(x1)
plt.show()

plt.plot(tups[0])
print(len(tups[0]))
plt.show()
