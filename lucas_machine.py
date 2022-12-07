import torch.nn as nn
import torch.nn.functional as F
import torch


class LucasMachine(nn.Module):
    def __init__(self, size, mid):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(size, mid),
            nn.ReLU(),
            nn.Linear(mid, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

