import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class LucasDataset(Dataset):
    def __init__(self, source):
        self.scaler = None
        self.df = self._preprocess(source)

    def _preprocess(self, source):
        self.scaler = MinMaxScaler()
        x_scaled = self.scaler.fit_transform(source[:, 0].reshape(-1, 1))
        source[:, 0] = np.squeeze(x_scaled)
        source[:, 1:] = 1 / (10 ** source[:, 1:])
        return source

    def unscale(self, value):
        return self.scaler.inverse_transform([[value]])[0][0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        this_x = self.get_x()[idx]
        soc = self.get_y()[idx]
        return torch.tensor(this_x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32)

    def get_y(self):
        return self.df[:,0]

    def get_x(self):
        return self.df[:,1:]

    def get_si(self, sif):
        return sif(self.get_x())


