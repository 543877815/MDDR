import shutil
import tarfile
import random
import h5py
import os

from attrdict import AttrDict

from utils import get_platform_path, is_image_file, rgb2ycbcr
from six.moves import urllib
import torch.utils.data as data
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import numpy as np
from tqdm import tqdm
import torch
import random


class DatasetFromFolders(data.Dataset):
    """
        Using this function, we must assure the size of HR are divide even by the size of LR, respectively.
    """

    def __init__(self, LR_dict: AttrDict, HR_dir: str, isTrain=False, config=None, transform=None,
                 target_transform=None):
        super(DatasetFromFolders, self).__init__()

        LR_dirs = [val for (key, val) in LR_dict.items()]

        LR_index_filenames = [os.listdir(LR_dir) for LR_dir in LR_dirs]
        HR_filenames = os.listdir(HR_dir)
        for LR_index_filename in LR_index_filenames:
            LR_index_filename.sort(key=lambda x: x[:-4])
        HR_filenames.sort(key=lambda x: x[:-4])

        if isTrain:
            for LR_index_filename in LR_index_filenames:
                LR_index_filename = LR_index_filename[:config.data_range]
            HR_filenames = HR_filenames[:config.data_range]
        else:
            for LR_index_filename in LR_index_filenames:
                LR_index_filename = LR_index_filename[config.data_range:]
            HR_filenames = HR_filenames[config.data_range:]

        self.LR_image_filenames = []
        for LR_dir in LR_dirs:
            self.LR_image_filenames.append([os.path.join(LR_dir, x) for x in HR_filenames if is_image_file(x)])
        self.HR_image_filenames = [os.path.join(HR_dir, x) for x in HR_filenames if is_image_file(x)]

        assert len(self.HR_image_filenames) == len(self.LR_image_filenames[0]), \
            'The number of HR images is not equal to the number of LR images!'

        self.config = config
        self.isTrain = isTrain
        self.repeat = config.repeat or 1
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        index = random.randint(0, len(self.LR_image_filenames) - 1)
        img = self.load_img(self.LR_image_filenames[index][item // self.repeat])
        target = self.load_img(self.HR_image_filenames[item // self.repeat])
        img, target = self.augment(LR_img=img, HR_img=target)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        if self.isTrain:
            img, target = self.get_patch(LR_img=img, HR_img=target)
        return index, img, target

    def __len__(self):
        if self.isTrain:
            return len(self.HR_image_filenames) * self.repeat
        else:
            return len(self.HR_image_filenames)

    # TODO test this function
    def augment(self, LR_img: PngImageFile, HR_img: PngImageFile) -> [PngImageFile, PngImageFile]:
        # flip
        flip_index = random.randint(0, len(self.config.flips) - 1)
        flip_type = self.config.flips[flip_index]
        if flip_type == 1:
            LR_img = LR_img.transpose(Image.FLIP_LEFT_RIGHT)
            HR_img = HR_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip_type == 2:
            LR_img = LR_img.transpose(Image.FLIP_TOP_BOTTOM)
            HR_img = HR_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip_index == 3:
            LR_img = LR_img.transpose(Image.FLIP_LEFT_RIGHT)
            HR_img = HR_img.transpose(Image.FLIP_LEFT_RIGHT)
            LR_img = LR_img.transpose(Image.FLIP_TOP_BOTTOM)
            HR_img = HR_img.transpose(Image.FLIP_TOP_BOTTOM)
        # rotation
        rotation_index = random.randint(0, len(self.config.rotations) - 1)
        angle = self.config.rotations[rotation_index]
        LR_img = LR_img.rotate(angle, expand=True)
        HR_img = HR_img.rotate(angle, expand=True)
        return LR_img, HR_img

    def get_patch(self, LR_img: torch.Tensor, HR_img: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        # TODO support multiple upscaleFactor
        scale_factor_index = random.randint(0, len(self.config.upscaleFactor) - 1)
        scale_factor_type = self.config.upscaleFactor[scale_factor_index]
        upscaleFactor = scale_factor_type
        width, height = LR_img.shape[1], LR_img.shape[2]
        size = self.config.img_size
        if self.config.same_size:
            tp = size
            ip = size
        else:
            tp = size
            ip = size // upscaleFactor
        ix = random.randrange(0, width - ip + 1)
        iy = random.randrange(0, height - ip + 1)
        if self.config.same_size:
            tx, ty = ix, iy
        else:
            tx, ty = upscaleFactor * ix, upscaleFactor * iy
        return LR_img[:, ix:ix + ip, iy:iy + ip], HR_img[:, tx:tx + tp, ty: ty + tp]

    def load_img(self, filepath):
        img = Image.open(filepath)
        if len(img.split()) == 1:
            return img
        img = img.convert('RGB')
        if self.config.color_space == 'RGB':
            return img
        elif self.config.color_space == 'YCbCr':
            img_ycrcb = rgb2ycbcr(np.array(img, dtype=np.uint8))
            if self.config.num_channels == 1:
                return Image.fromarray(img_ycrcb[:, :, 0])
            else:
                return Image.fromarray(img_ycrcb)
        else:
            raise Exception("the color space does not exist")


