import PIL.Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
import dwt
import numpy

class LucasDataset(Dataset):
    def __init__(self, is_train=True):
        self.preload = False
        self.dump = False
        self.DWT = False
        self.is_train = is_train
        self.csv_file_location = "data/lucas-many.csv"
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
            self.x = dwt.transform(self.df[self.df.columns[1:]].values)
        else:
            self.x = self.df[self.df.columns[1:]].values
        self.y = self.df[self.df.columns[0]].values



        s471 = self.x[:,0].reshape(-1,1)
        blue478 = self.x[:,1].reshape(-1,1)
        s500 = self.x[:,2].reshape(-1,1)
        s530 = self.x[:,3].reshape(-1,1)
        green546 = self.x[:,4].reshape(-1,1)
        b3_560 = self.x[:,5].reshape(-1,1)
        s590 = self.x[:,6].reshape(-1,1)
        s620 = self.x[:,7].reshape(-1,1)
        red659 = self.x[:,8].reshape(-1,1)
        b4_665 = self.x[:,9].reshape(-1,1)
        s709 = self.x[:,10].reshape(-1,1)
        b8_842_nir = self.x[:,11].reshape(-1,1)
        s844 = self.x[:,12].reshape(-1,1)
        s1001 = self.x[:,13].reshape(-1,1)
        s1064 = self.x[:,14].reshape(-1,1)
        s1094 = self.x[:,15].reshape(-1,1)
        s1104 = self.x[:,16].reshape(-1,1)
        s1114 = self.x[:,18].reshape(-1,1)
        s1185 = self.x[:,18].reshape(-1,1)
        s1245 = self.x[:,19].reshape(-1,1)
        s1316 = self.x[:,20].reshape(-1,1)
        s1498 = self.x[:,21].reshape(-1,1)
        s1558 = self.x[:,22].reshape(-1,1)
        s1588 = self.x[:,23].reshape(-1,1)
        s1790 = self.x[:,24].reshape(-1,1)
        s1982 = self.x[:,25].reshape(-1,1)
        s1992 = self.x[:,26].reshape(-1,1)
        s2052 = self.x[:,27].reshape(-1,1)
        s2102 = self.x[:,28].reshape(-1,1)
        s2103 = self.x[:,29].reshape(-1,1)
        s2150 = self.x[:,30].reshape(-1,1)
        s2163 = self.x[:,31].reshape(-1,1)
        s2355 = self.x[:,32].reshape(-1,1)

        inv_blue_2 = 1 / (blue478 ** 2)
        self.new_x = numpy.concatenate((self.x, inv_blue_2), axis=1)

    def _preprocess(self, df):
        df = self.__scale__(df)
        for col in df.columns[1:]:
            df[col] = (1/(10**df[[col]].values.astype(float)))
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
        this_new_x = self.new_x[idx]
        return torch.tensor(this_x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32),\
            torch.tensor(this_new_x, dtype=torch.float32)

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_new_x(self):
        return self.new_x


if __name__ == "__main__":
    cid = LucasDataset()
    print(cid.unscale(0.5))
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)

