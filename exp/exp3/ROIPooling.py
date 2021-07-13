import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.nn import init

class Classifier(nn.Module):
    def __init__(self, n_classes=3, in_channels=3, RoIPooling_shape: list = [7, 7], img_shape=[3, 128, 128],
                 channels: int = 64, grow_channels: int = 32, scale_ratio: float = 0.2):
        super(Classifier, self).__init__()

        self.n_classes = n_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(256),
            nn.Conv2d(256, 1, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.MaxPooling = nn.AdaptiveAvgPool2d((4, 4))
        self.linear = nn.Linear(16, n_classes)

    def forward(self, x):
        out = self.model(x)
        out = self.MaxPooling(out)
        out = out.view(out.shape[0], 16)
        return self.linear(out)
