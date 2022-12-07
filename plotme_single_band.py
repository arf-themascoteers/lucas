import numpy as np
import matplotlib.pyplot as plt

NP_FILE = "nps/one_band.npy"
def plot_please():
    array = np.load(NP_FILE)
    x_ticks = np.array(list(range(0, array.shape[0], 100)))
    xticklabels = x_ticks + 400
    plt.xticks(x_ticks, xticklabels, rotation='vertical')
    plt.xlabel("Wavelength")
    plt.ylabel("R^2")
    plt.subplots_adjust(bottom=0.15)
    plt.plot(array)
    plt.show()
    print("done")

if __name__ == "__main__":
    plot_please()
    array = np.load(NP_FILE)
    indices = np.argsort(array)
    for i in range(10):
        index = indices[len(indices) - 1 - i]
        print(index+400, array[index])
    print(max(array))