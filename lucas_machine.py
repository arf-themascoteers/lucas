import torch.nn as nn
import torch.nn.functional as F
import torch


class LucasMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_net = nn.Sequential(
            nn.Linear(132,1)
        )

    def forward(self, x):
        x = self.band_net(x)
        return x

