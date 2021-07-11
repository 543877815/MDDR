import sys
import os
from utils import get_config
from torch.utils.data import DataLoader
from attrdict import AttrDict
from options import args
from utils import get_config
from dataset import DatasetFromFolders
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.nn as nn
from model.solver import MDDRTrainer


def get_dataset(config: AttrDict):
    data_flist = AttrDict(config.data_flist[config.platform])
    train_LR_dir, train_HR_dir = data_flist.train_LR_dir, data_flist.train_HR_dir
    test_LR_dir, test_HR_dir = data_flist.test_LR_dir, data_flist.test_HR_dir

    # data transform
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = DatasetFromFolders(LR_dict=train_LR_dir, HR_dir=train_HR_dir, isTrain=True, transform=img_transform,
                                   target_transform=target_transform, config=config)
    test_set = DatasetFromFolders(LR_dict=test_LR_dir, HR_dir=test_HR_dir, isTrain=False, transform=img_transform,
                                  target_transform=target_transform, config=config)

    return train_set, test_set


def get_trainer(config, train_loader, test_loader, device=None):
    model = MDDRTrainer(config=config, train_loader=train_loader, test_loader=test_loader, device=device)
    return model


if __name__ == '__main__':
    # get configuration
    config = get_config(args)

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    if not config.distributed:
        device = torch.device(
            "cuda:{}".format(config.gpu[0]) if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device("cuda", args.local_rank)

    # get dataset
    train_set, test_set = get_dataset(config=config)

    sampler = None
    local_rank = None
    if config.distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        config.device = torch.device("cuda", local_rank)
        sampler = DistributedSampler(dataset=train_set, shuffle=True)

    train_loader = DataLoader(dataset=train_set, batch_size=config.batch_size, shuffle=sampler is None,
                              pin_memory=True, num_workers=config.num_workers, drop_last=False, sampler=sampler)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    trainer = get_trainer(config=config, train_loader=train_loader, test_loader=test_loader, device=device)

    # get models
    # trainer = get_trainer(config, train_loader, test_loader, device)
    if config.distributed and len(config.gpu) > 1:
        # print(args.rank, args.world_size, args.local_rank, config.gpu)
        # assert args.rank is None and args.world_size is None, \
        #     'When --distributed is enabled (default) the rank and ' + \
        #     'world size can not be given as this is set up automatically. ' + \
        #     'Use --distributed 0 to disable automatic setup of distributed training.'
        trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model, output_device=args.local_rank,
                                                                  device_ids=[args.local_rank])
    elif len(config.gpu) > 1:
        trainer.model = nn.DataParallel(trainer.model, device_ids=config.gpu)
    trainer.run()
