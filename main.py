import torch
import train
import test
import ds_manager
import train
import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    dm = ds_manager.DSManager()
    train_ds = dm.get_train_ds()
    test_ds = dm.get_test_ds()

    reg = train.train(device, train_ds)
    r2 = test.test(device, test_ds, reg)

    print(r2)