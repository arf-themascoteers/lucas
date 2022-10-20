import PIL.Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
import dwt


class LucasDataset(Dataset):
    def __init__(self, is_train=True):
        self.preload = False
        self.dump = True
        self.DWT = False
        self.is_train = is_train
        self.csv_file_location = "data/lucas.csv"
        self.work_csv_file_location_train = "data/train.csv"
        self.work_csv_file_location_test = "data/test.csv"
        self.scaler = None
        if self.preload:
            if self.is_train:
                self.df = pd.read_csv(self.work_csv_file_location_train)
            else:
                self.df = pd.read_csv(self.work_csv_file_location_test)
            self.df = self._preprocess(self.df)
        else:
            self.df = pd.read_csv(self.csv_file_location)
            self.df = self.df.drop(columns=["lc1","lu1"])
            #self.df = self.df.loc[self.df['oc'] <= 40]
            train, test = model_selection.train_test_split(self.df, test_size=0.2)
            self.df = train
            if not self.is_train:
                self.df = test

            self.df = self._preprocess(self.df)

            if self.dump:
                if self.is_train:
                    self.df.to_csv(self.work_csv_file_location_train, index=False)
                else:
                    self.df.to_csv(self.work_csv_file_location_test, index=False)

        if self.DWT:
            self.x = dwt.transform(self.df[self.df.columns[11:]].values)
        else:
            self.x = self.df[self.df.columns[11:]].values
            #self.x = self.df[self.df.columns[[2560,2850,1580,2844,450,1761,3508,1068,714,674,868,1823,2135,1216,3215,1133,1973,1317,2232,2956,1101]]].values

        self.aux = self.df[self.df.columns[2:11]].values
        self.y = self.df[self.df.columns[1]].values

    def _preprocess(self, df):
        df = self.__scale__(df)
        return df

    def __scale__(self, df):
        for col in df.columns[1:11]:
            if col == "lc1" or col == "lu1":
                continue
            df, x_scaler = self.__scale_col__(df, col)
            if col == "oc":
                self.scaler = x_scaler

        # x = df[df.columns[11:]].values.astype(float)
        # nrows = x.shape[0]
        # ncols = x.shape[1]
        # x = x.reshape((nrows * ncols, 1))
        # scaler = MinMaxScaler()
        # x_scaled = scaler.fit_transform(x)
        # x_scaled = x_scaled.reshape((nrows, ncols))
        # df[df.columns[11:]] = x_scaled

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
        this_aux = self.aux[idx]
        soc = self.y[idx]
        return torch.tensor(this_x, dtype=torch.float32), torch.tensor(this_aux, dtype=torch.float32),\
               torch.tensor(soc, dtype=torch.float32)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_aux(self):
        return self.aux


if __name__ == "__main__":
    cid = LucasDataset()
    print(cid.unscale(0.5))
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)

