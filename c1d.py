import torch
import torch.nn.functional as F
from lucas_dataset import LucasDataset
import torch.nn as nn
from torch.utils.data import DataLoader
from lucas_1d import Lucas1D
from sklearn.metrics import r2_score


def train(device):
    batch_size = 5000
    cid = LucasDataset(is_train=True)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=True)
    model = Lucas1D()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss(reduction='mean')
    num_epochs = 200
    n_batches = int(len(cid)/batch_size) + 1
    batch_number = 0
    loss = None

    for epoch in range(num_epochs):
        batch_number = 0
        for (x, aux, y) in dataloader:
            x = x.to(device)
            aux = aux.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x, aux)
            y_hat = y_hat.reshape(-1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_number += 1
            print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    print("Train done")
    torch.save(model, 'models/soc.h5')

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
    print(f"Loss {loss_cum:.2f}")
    print(f"R^2 {r2_score(actuals, predicteds):.2f}")

    for i in range(5):
        print(f"{actuals[i]:.3f}\t\t{predicteds[i]:.3f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Train...")
train(device)
print("Test...")
test(device)
