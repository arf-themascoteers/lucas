import torch
from torch.utils.data import DataLoader
from lucas_machine import LucasMachine
import time


def train(device, ds):
    batch_size = 250
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    x_size = ds.get_x().shape[1]
    mid = 2
    if x_size < 10:
        mid = 500
        batch_size = 500
    model = LucasMachine(size = x_size, mid = mid)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    criterion = torch.nn.MSELoss(reduction='sum')
    num_epochs = 300
    n_batches = int(len(ds)/batch_size) + 1
    batch_number = 0
    loss = None
    start = time.time()
    for epoch in range(num_epochs):
        batch_number = 0
        for (x, y) in dataloader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            y_hat = y_hat.reshape(-1)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_number += 1
            #print(f'Epoch:{epoch + 1} (of {num_epochs}), Batch: {batch_number} of {n_batches}, Loss:{loss.item():.6f}')

    #print("Train done")
    end = time.time()
    required = end - start
    #print(f"Train seconds: {required}")
    torch.save(model, 'models/soc.h5')
    return model

#
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     train(device)