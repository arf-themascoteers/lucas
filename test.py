import torch
from lucas_dataset import LucasDataset
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

def test(device):
    batch_size = 10
    cid = LucasDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/soc.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0

    loss_cum = 0
    itr = 0
    actuals = []
    predicteds = []

    # print(f"Actual SOC\t\t\tPredicted SOC")
    for (x, aux, y) in dataloader:
        x = x.to(device)
        aux = aux.to(device)
        y = y.to(device)
        y_hat = model(x, aux)
        y_hat = y_hat.reshape(-1)
        loss = criterion(y_hat, y)
        itr = itr+1
        loss_cum = loss_cum + loss.item()

        for i in range(y_hat.shape[0]):
            actuals.append(y[i].detach().item())
            predicteds.append(y_hat[i].detach().item())

    loss_cum = loss_cum / itr
    print(f"Loss {loss_cum:.4f}")
    print(f"R^2 {r2_score(actuals, predicteds):.4f}")

    # for i in range(10):
    #     print(f"{actuals[i]:.3f}\t\t{predicteds[i]:.3f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
