import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class LucasDataset(Dataset):
    def __init__(self, source):
        self.source = source
        self.scaler = None
        self.df = self._preprocess(source)
        self.x = self.df[self.df.columns[1:]].values
        self.y = self.df[self.df.columns[0]].values

    def _preprocess(self, df):
        df = self.__scale__(df)
        for col in df.columns[1:]:
            vals = df[[col]].values.astype(float)
            df[col] = (1/(10**vals))
        return df

    def __scale__(self, df):
        df, self.scaler = self.__scale_col__(df, "soc")
        return df

    def __scale_col__(self, df, col):
        x = df[[col]].values.astype(float)
        a_scaler = MinMaxScaler()
        x_scaled = a_scaler.fit_transform(x)
        df[col] = x_scaled
        return df, a_scaler

    def unscale(self, value):
        return self.scaler.inverse_transform([[value]])[0][0]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        this_x = self.x[idx]
        soc = self.y[idx]
        return torch.tensor(this_x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_si(self, sif):
        return sif(self.x)


if __name__ == "__main__":
    cid = LucasDataset()
    print(cid.unscale(0.5))
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)

