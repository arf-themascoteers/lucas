import torch.nn as nn
import torch.nn.functional as F
import torch


class LucasMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_net = nn.Sequential(
            nn.Conv1d(1,4,10),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=10),
            nn.Conv1d(4,16,10),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=10),
            nn.Conv1d(16, 32, 10),
            nn.LeakyReLU(),
            nn.AvgPool1d(kernel_size=10),
            nn.Flatten(),
            nn.Linear(96,1)
        )

    def forward(self, x):
        x = x.reshape(x.shape[0],1,x.shape[1])
        x = self.band_net(x)
        return x

