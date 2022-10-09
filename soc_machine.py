import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torchvision.models import resnet50, ResNet50_Weights


class SOCMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_net = nn.Sequential(
            nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.band_net(x)
        return x

