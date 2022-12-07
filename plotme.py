import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

def plot_please():
    array = np.load("nps/ndi4.npy")
    mask = np.zeros_like(array)
    mask[np.tril_indices_from(mask)] = True
    x_ticks = np.array(list(range(0,array.shape[0], 200)))
    xticklabels = x_ticks + 400
    with sns.axes_style("white"):
        ax = sns.heatmap(array, mask=mask, vmax=0.5, square=True,
                         cmap="mako")
        #ax.set_xticks(range(len(x_indices)))
        #ax.set_xticklabels(x_indices)
        #ax.set_yticklabels(x_indices)
        #ax.set_xticks([100,200])
        plt.xlabel("B1")
        plt.ylabel("B2")
        plt.xticks(x_ticks, xticklabels)
        plt.yticks(x_ticks, xticklabels)
        plt.subplots_adjust(bottom=0.15)
        ax.invert_yaxis()
    plt.show()
    print("done")

if __name__ == "__main__":
    plot_please()