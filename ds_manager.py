import lucas_dataset
from sklearn import model_selection
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class DSManager:
    def __init__(self, size = "full", btype="absorbance", ctype=None):
        csv_file_location = "data/min.csv"
        if size == "min":
            csv_file_location = "data/min.csv"
        df = pd.read_csv(csv_file_location)
        npdf = df.to_numpy()
        npdf = self._preprocess(npdf, btype, ctype)
        self.train, self.test = model_selection.train_test_split(npdf, test_size=0.2, random_state=1)
        self.test_ds = None
        self.train_ds = None

    def get_test_ds(self):
        if self.test_ds is not None:
            return self.test_ds

        self.test_ds = lucas_dataset.LucasDataset(self.test)
        return self.test_ds

    def get_train_ds(self):
        if self.train_ds is not None:
            return self.train_ds

        self.train_ds = lucas_dataset.LucasDataset(self.train)
        return self.train_ds

    def _preprocess(self, source, btype, ctype):
        self.scaler = MinMaxScaler()
        x_scaled = self.scaler.fit_transform(source[:, 0].reshape(-1, 1))
        source[:, 0] = np.squeeze(x_scaled)
        if btype == "reflectance" or ctype is not None:
            source[:, 1:] = 1 / (10 ** source[:, 1:])

        if ctype is not None:
            blue = self._get_wavelength(source, 478)
            green = self._get_wavelength(source, 546)
            red = self._get_wavelength(source, 659)
            source = np.concatenate((source[:,0], blue, green, red), axis=1)

        return source

    def _get_wavelength(self, data, wl):
        index = (wl - 400)*2
        return data[wl]

if __name__ == "__main__":
    dm = DSManager()
    train_ds = dm.get_train_ds()
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)