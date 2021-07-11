import logging
import sys
import time

import yaml
from attrdict import AttrDict
import platform
import os
from os import path as osp
import numpy as np


def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    https://github.com/xinntao/BasicSR/blob/master/scripts/data_preparation/create_lmdb.py
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def prepare_keys_div2k(folder_path):
    """Prepare image path list and keys for DIV2K dataset.
    Args:
        folder_path (str): Folder path.
    Returns:
        list[str]: Image path list.
        list[str]: Key list.
    """
    print('Reading image path list ...')
    img_path_list = sorted(list(scandir(folder_path, suffix='png', recursive=False)))
    keys = [img_path.split('.png')[0] for img_path in sorted(img_path_list)]

    return img_path_list, keys


def get_config(args):
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
            for arg in vars(args):
                if args.config_priority == 'args':
                    config[arg] = getattr(args, arg)
                elif arg not in config.keys():
                    config[arg] = getattr(args, arg)
            config = AttrDict(config)
        except yaml.YAMLError as exc:
            config = None
            print(exc)
    return config


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def print_options(opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for key in sorted(opt):
        if key == 'data_flist':
            for platform in opt[key]:
                for item in opt[key][platform]:
                    message += '{:>25}.{}.{}: {:<30}\n'.format(str(key), str(platform), str(item), str(opt[key][platform][item]))
        else:
            message += '{:>25}: {:<30}\n'.format(str(key), str(opt[key]))
    message += '----------------- End -------------------'
    return message


TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T


# return the formatted time
def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds * 1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + 's'
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + 'ms'
        time_index += 1
    if output == '':
        output = '0ms'
    return output


def progress_bar(current, total, msg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpeg', '.jpg', '.bmp', '.JPEG'])


def get_platform_path(config=None):
    system = platform.system()
    data_dir, model_dir, checkpoint_dir, log_dir, dirs = '', '', '', '', []
    if config and config.use_relative:
        checkpoint_dir = 'checkpoint/'
        model_dir = 'model/'
        log_dir = 'log/'
    else:
        if system == 'Windows':
            drive, common_dir = 'F', 'cache'
            data_dir = '{}:/{}/data'.format(drive, common_dir)
            model_dir = '{}:/{}/model'.format(drive, common_dir)
            checkpoint_dir = '{}:/{}/checkpoint'.format(drive, common_dir)
            log_dir = '{}:/{}/log'.format(drive, common_dir)
            dirs = [data_dir, model_dir, checkpoint_dir, log_dir]

        elif system == 'Linux':
            common_dir = '/data'
            data_dir = '{}/data'.format(common_dir)
            model_dir = '{}/model'.format(common_dir)
            checkpoint_dir = '{}/checkpoint'.format(common_dir)
            log_dir = '{}/log'.format(common_dir)
            dirs = [data_dir, model_dir, checkpoint_dir, log_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    return data_dir, model_dir, checkpoint_dir, log_dir


def shave(pred, gt, shave_border):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    return pred, gt


# / 255
OrigT = np.array(
    [[65.481, 128.553, 24.966],
     [-37.797, -74.203, 112.0],
     [112.0, -93.786, -18.214]])

OrigOffset = np.array([16, 128, 128])

# OrigT_inv = np.array([[0.00456621,  0.,  0.00625893],
#           [0.00456621, -0.00153632, -0.00318811],
#           [0.00456621,  0.00791071,  0.]])
OrigT_inv = np.linalg.inv(OrigT)


def rgb2ycbcr(rgb_img):
    if rgb_img.shape[2] == 1:
        return rgb_img
    if rgb_img.dtype == float:
        T = 1.0 / 255.0
        offset = 1 / 255.0
    elif rgb_img.dtype == np.uint8:
        T = 1.0 / 255.0
        offset = 1.0
    elif rgb_img.dtype == np.uint16:
        T = 257.0 / 65535.0
        offset = 257.0
    else:
        raise Exception('the dtype of image does not support')
    T = T * OrigT
    offset = offset * OrigOffset
    ycbcr_img = np.zeros(rgb_img.shape, dtype=float)
    for p in range(rgb_img.shape[2]):
        ycbcr_img[:, :, p] = T[p, 0] * rgb_img[:, :, 0] + T[p, 1] * rgb_img[:, :, 1] + T[p, 2] * rgb_img[:, :, 2] + \
                             offset[p]
    return np.array(ycbcr_img, dtype=rgb_img.dtype)


def ycbcr2rgb(ycbcr_img):
    if ycbcr_img.shape[2] == 1:
        return ycbcr_img
    if ycbcr_img.dtype == float:
        T = 255.0
        offset = 1.0
    elif ycbcr_img.dtype == np.uint8:
        T = 255.0
        offset = 255.0
    elif ycbcr_img.dtype == np.uint16:
        T = 65535.0 / 257.0
        offset = 65535.0
    else:
        raise Exception('the dtype of image does not support')
    T = T * OrigT_inv
    offset = offset * np.matmul(OrigT_inv, OrigOffset)
    rgb_img = np.zeros(ycbcr_img.shape, dtype=float)
    for p in range(rgb_img.shape[2]):
        rgb_img[:, :, p] = T[p, 0] * ycbcr_img[:, :, 0] + T[p, 1] * ycbcr_img[:, :, 1] + T[p, 2] * ycbcr_img[:, :, 2] - \
                           offset[p]
    return np.array(rgb_img.clip(0, 255), dtype=ycbcr_img.dtype)
