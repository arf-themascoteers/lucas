import PIL.Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection


class LucasDataset(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.csv_file_location = "data/lucasmid.csv"
        self.work_csv_file_location = "data/work.csv"
        self.scaler = None
        self.df = pd.read_csv(self.csv_file_location)
        self.df = self.df.drop(columns=["lc1","lu1"])
        self.df = self.df.loc[self.df['oc'] <= 40]
        train, test = model_selection.train_test_split(self.df, test_size=0.2)
        self.df = train
        if not self.is_train:
            self.df = test

        self.df = self._preprocess(self.df)
        self.df.to_csv(self.work_csv_file_location, index = False)

    def _preprocess(self, df):
        self.__scale__(df)
        return df

    def __scale__(self, df):
        for col in df.columns[1:11]:
            if col == "lc1" or col == "lu1":
                continue
            df, x_scaler = self.__scale_col__(df, col)
            if col == "oc":
                self.scaler = x_scaler

        x = df[df.columns[11:]].values.astype(float)
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        df[df.columns[11:]] = x_scaled

        return df

    def __scale_col__(self, df, col):
        x = df[[col]].values.astype(float)
        a_scaler = MinMaxScaler()
        x_scaled = a_scaler.fit_transform(x)
        df[col] = x_scaled
        return df, x_scaled

    def unscale(self, values):
        values = [[i] for i in values]
        values = self.scaler.inverse_transform(values)
        values = [i[0] for i in values]
        return values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        soc = row["oc"]
        x = list(row[11:])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(soc, dtype=torch.float32)

    def get_x(self):
        return self.df[self.df.columns[11:]].values
        #return self.df.iloc[0:100, 2:11].values

    def get_y(self):
        return self.df[self.df.columns[1]].values
        #return self.df.iloc[0:100, 1].values


if __name__ == "__main__":
    cid = LucasDataset()
    dataloader = DataLoader(cid, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)

