import torchvision
import torch
from torchvision.datasets.folder import make_dataset, pil_loader
import os
import random


class myImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, HR=None, transform=None, target_transform=None):
        super(myImageFolder, self).__init__(root)
        self.root = root
        classes, class_to_idx = self._find_classes(self.root)
        extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
        samples = make_dataset(self.root, class_to_idx, extensions)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.loader = pil_loader
        self.HR_path = HR
        self.img_size = 128
        self.same_size = True
        self.transform = transform
        self.target_transform = target_transform

    def get_patch(self, LR_img: torch.Tensor, HR_img: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        # TODO support multiple upscaleFactor
        upscaleFactor = 2
        width, height = LR_img.shape[1], LR_img.shape[2]
        size = self.img_size
        if self.same_size:
            tp = size
            ip = size
        else:
            tp = size
            ip = size // upscaleFactor
        ix = random.randrange(0, width - ip + 1)
        iy = random.randrange(0, height - ip + 1)
        if self.same_size:
            tx, ty = ix, iy
        else:
            tx, ty = upscaleFactor * ix, upscaleFactor * iy
        return LR_img[:, ix:ix + ip, iy:iy + ip], HR_img[:, tx:tx + tp, ty: ty + tp]

    def __getitem__(self, index):
        path, target = self.samples[index]
        name = os.path.basename(path)
        HR_path = os.path.join(self.HR_path, name)

        LR = self.loader(path)
        HR = self.loader(HR_path)

        if self.transform:
            LR, HR = self.transform(LR), self.transform(HR)
        if self.target_transform:
            target = self.target_transform(target)

        LR_patch, HR_patch = self.get_patch(LR, HR)
        return LR_patch, HR_patch, target
