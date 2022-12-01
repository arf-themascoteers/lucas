import lucas_dataset
from sklearn import model_selection
import pandas as pd
from torch.utils.data import DataLoader


class DSManager:
    def __init__(self):
        csv_file_location = "data/miner.csv"
        df = pd.read_csv(csv_file_location)
        npdf = df.to_numpy()
        self.train, self.test = model_selection.train_test_split(npdf, test_size=0.2)
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


if __name__ == "__main__":
    dm = DSManager()
    train_ds = dm.get_train_ds()
    dataloader = DataLoader(train_ds, batch_size=1, shuffle=True)
    for x, soc in dataloader:
        print(x)
        print(x.shape[1])
        print(soc)
        exit(0)