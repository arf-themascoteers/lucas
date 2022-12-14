import torch
from lucas_dataset import LucasDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import time

def test(device, ds, model):
    batch_size = 30000
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss(reduction='mean')
    #model = torch.load("models/soc.h5")
    model.eval()
    model.to(device)


    loss_cum = 0
    itr = 0

    start = time.time()
    r2 = 0
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        y_hat = y_hat.reshape(-1)
        loss = criterion(y_hat, y)
        itr = itr+1
        loss_cum = loss_cum + loss.item()

        r2 = r2_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        loss_cum = loss_cum / itr
        #print(f"Loss {loss.item():.4f}")
        #print(f"R^2 {r2:.4f}")

    end = time.time()
    required = end - start
    #print(f"Test seconds: {required}")
    return r2

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     test(device)
