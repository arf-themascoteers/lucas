import torch.nn as nn
import torch.nn.functional as F
import torch


class LucasMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4200, 3000),
            nn.ReLU(),
            nn.Linear(3000, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

