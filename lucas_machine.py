import torch.nn as nn
import torch.nn.functional as F
import torch


class LucasMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_net = nn.Sequential(
            nn.Linear(132, 100),
            nn.ReLU(),
            nn.Linear(100,1)
        )

    def forward(self, x, aux):
        #x = torch.cat((x,aux), dim=1)
        x = self.band_net(x)
        return x

