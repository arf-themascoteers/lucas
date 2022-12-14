from lucas_dataset import LucasDataset
from sklearn import model_selection
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import colorsys
from sklearn.model_selection import KFold
import time
import colour



class DSManager:
    def __init__(self, size = "full", btype="absorbance", ctype=None, si=[], si_only = False, name=None):
        PRELOAD = True
        self.name = name
        if self.name is None:
            epoch_time = str(int(time.time()))
            self.name = f"{btype}_{ctype}_{epoch_time}"

        if PRELOAD:
            npdf = np.load(f"nps/npdf.npy")
        else:
            csv_file_location = "data/lucas.csv"
            if size == "min":
                csv_file_location = "data/min.csv"
            df = pd.read_csv(csv_file_location)
            npdf = df.to_numpy()
            np.save("nps/npdf.npy", npdf)

        npdf = self._preprocess(npdf, btype, ctype, si, si_only)
        train, test = model_selection.train_test_split(npdf, test_size=0.2, random_state=1)
        self.full_data = np.concatenate((train, test), axis=0)
        self.full_ds = LucasDataset(npdf)
        self.train_ds = LucasDataset(train)
        self.test_ds = LucasDataset(test)

    def get_test_ds(self):
        return self.test_ds

    def get_train_ds(self):
        return self.train_ds

    def get_full_ds(self):
        return self.full_ds

    def get_name(self):
        return self.name

    def get_10_folds(self):
        kf = KFold(n_splits=10)
        for i, (train_index, test_index) in enumerate(kf.split(self.full_data)):
            train_data = self.full_data[train_index]
            test_data = self.full_data[test_index]
            yield LucasDataset(train_data), LucasDataset(test_data)

    def _preprocess(self, absorbance, btype, ctype, si, si_only):
        data = absorbance.copy()
        self.scaler = MinMaxScaler()
        x_scaled = self.scaler.fit_transform(data[:, 0].reshape(-1, 1))
        data[:, 0] = np.squeeze(x_scaled)

        if si_only:
            data = data[:,0:1]
        else:
            if ctype is not None:
                blue = self.get_blue(absorbance).reshape(-1, 1)
                green = self.get_green(absorbance).reshape(-1, 1)
                red = self.get_red(absorbance).reshape(-1, 1)
                data = np.concatenate((data[:,0:1], blue, green, red), axis=1)

            if btype == "reflectance":
                data = self.get_reflectance(data)

            if ctype == "hsv":
                data = self.get_hsv(data)

            if ctype == "rgbhsv":
                dest = self.get_hsv(data)
                data = np.concatenate((data, dest[:,1:]), axis=1)

            if ctype == "bigxyz":
                dest = self.get_bigxyz(data)
                data = np.concatenate((data, dest[:,1:]), axis=1)

        if len(si) > 0:
            reflectance = self.get_reflectance(absorbance)
            for spectral_index in si:
                si_vals = self.get_si(reflectance, spectral_index).reshape(-1, 1)
                data = np.concatenate((data, si_vals), axis=1)

        return data

    def get_reflectance(self, source):
        data = source.copy()
        data[:, 1:] = 1 / (10 ** data[:, 1:])
        return data

    def get_hsv(self, rgb):
        dest = rgb.copy()
        for i in range(rgb.shape[0]):
            b, g, r = rgb[i, 1], rgb[i, 2], rgb[i, 3]
            (h, s, v) = colorsys.rgb_to_hsv(r, g, b)
            dest[i, 1], dest[i, 2], dest[i, 3] = v, s, h
        return dest

    def _get_wavelength(self, data, wl):
        index = (wl - 400)*2
        return data[:,index]

    def get_si(self, data, spectral_index):
        if spectral_index == "soci":
            return self.get_soci(data)
        if spectral_index == "ibs":
            return self.get_ibs(data)

        return None

    def get_soci(self, data):
        blue = self.get_blue(data)
        green = self.get_green(data)
        red = self.get_red(data)
        return (blue)/(red*green)

    def get_ibs(self, data):
        blue = self.get_blue(data)
        return 1/(blue**2)

    def get_blue(self, source):
        return self._get_wavelength(source, 478)

    def get_green(self, source):
        return self._get_wavelength(source, 546)

    def get_red(self, source):
        return self._get_wavelength(source, 659)

    def get_bigxyz(self, rgb):
        dest = rgb.copy()
        for i in range(rgb.shape[0]):
            rgb_array = [rgb[i, 3], rgb[i, 2], rgb[i, 1]]
            XYZ = colour.sRGB_to_XYZ(rgb)
            dest[i, 3], dest[i, 2], dest[i, 1] = XYZ[0], XYZ[1], XYZ[2]
        return dest


if __name__ == "__main__":
    dm = DSManager()
    train_ds = dm.get_train_ds()
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)