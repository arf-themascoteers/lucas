import lucas_dataset
from sklearn import model_selection
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import colorsys


class DSManager:
    def __init__(self, size = "full", btype="absorbance", ctype=None):
        PRELOAD = True

        if PRELOAD:
            npdf = np.load("nps/npdf.npy")
        else:
            csv_file_location = "data/lucas.csv"
            if size == "min":
                csv_file_location = "data/min.csv"
            df = pd.read_csv(csv_file_location)
            npdf = df.to_numpy()
            np.save("nps/npdf.npy", npdf)

        npdf = self._preprocess(npdf, btype, ctype)
        train, test = model_selection.train_test_split(npdf, test_size=0.2, random_state=1)

        self.test_ds = lucas_dataset.LucasDataset(test)
        self.train_ds = lucas_dataset.LucasDataset(train)

    def get_test_ds(self):
        return self.test_ds

    def get_train_ds(self):
        return self.train_ds

    def _preprocess(self, source, btype, ctype):
        self.scaler = MinMaxScaler()
        x_scaled = self.scaler.fit_transform(source[:, 0].reshape(-1, 1))
        source[:, 0] = np.squeeze(x_scaled)

        if ctype is not None:
            blue = self._get_wavelength(source, 478).reshape(-1,1)
            green = self._get_wavelength(source, 546).reshape(-1,1)
            red = self._get_wavelength(source, 659).reshape(-1,1)
            source = np.concatenate((source[:,0:1], blue, green, red), axis=1)

        if btype == "reflectance" or ctype is not None:
            source[:, 1:] = 1 / (10 ** source[:, 1:])

        if ctype == "hsv":
            source = self.get_hsv(source)

        if ctype == "rgbhsv":
            dest = self.get_hsv(source)
            source = np.concatenate((source, dest[:,1:]), axis=1)

        return source

    def get_hsv(self, source):
        dest = np.zeros_like(source)
        for i in range(source.shape[0]):
            b, g, r = source[i, 1], source[i, 2], source[i, 3]
            (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
            dest[i, 1], dest[i, 2], dest[i, 3] = v, s, h
        return dest

    def _get_wavelength(self, data, wl):
        index = (wl - 400)*2
        return data[:,index]


if __name__ == "__main__":
    dm = DSManager()
    train_ds = dm.get_train_ds()
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)