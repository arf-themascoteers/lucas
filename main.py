import torch
import train
import test
import lucas_dataset
import train
import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if __name__ == "__main__":
    train_ds = lucas_dataset.LucasDataset(is_train=True)
    test_ds = lucas_dataset.LucasDataset(is_train=False)

    reg = train.train(device, train_ds)
    without_si = test.test(device, test_ds, reg)

    reg = train.train(device, train_ds, use_new_x=True)
    with_si = test.test(device, test_ds, reg, use_new_x=True)

    print(without_si, with_si)
