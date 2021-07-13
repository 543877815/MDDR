from torchvision.datasets.folder import pil_loader
import torchvision
import torch
import os
import random

class myDataset(torchvision.datasets.DatasetFolder):
    def __init__(self, LR, HR, transform=None):
        super(myDataset, self).__init__()
