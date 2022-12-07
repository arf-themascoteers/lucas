import ds_manager
import torch
import train
import test
import os


def run_plz():
    os.chdir("../../")
    dm = ds_manager.DSManager(btype="reflectance", ctype="rgbhsv")
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train.train(device, train_ds)
    r2 = test.test(device, test_ds, model)

    print(r2)


if __name__ == "__main__":
    run_plz()