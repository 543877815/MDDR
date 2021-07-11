import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.nn import init


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class RDB(nn.Module):
    r"""
    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, grow_channels: int = 32, scale_ratio: float = 0.2):
        super(RDB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(channels + 4 * grow_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))

        out = torch.add(conv5 * self.scale_ratio, x)

        return out


# class Classifier(nn.Module):
#     def __init__(self, n_classes: int = 3, RoIPooling_shape: list = [7, 7], img_shape=[3, 128, 128],
#                  channels: int = 64, grow_channels : int = 32, scale_ratio: float = 0.2):
#         super(Classifier, self).__init__()
#
#         self.head = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, stride=1, padding=1)
#         self.RDB = RDB(channels, grow_channels, scale_ratio)
#         self.end = nn.Conv2d(in_channels=channels, out_channels=n_classes, kernel_size=3, stride=1, padding=1)
#         self.MaxPool2d = nn.MaxPool2d(img_shape[-1], stride=1)
#         # self.RoIPooling = nn.AdaptiveMaxPool2d(RoIPooling_shape)
#         # self.RoIPooling_shape = RoIPooling_shape
#         # self.Linear = nn.Sequential(
#         #     nn.Linear(in_features=RoIPooling_shape[0] * RoIPooling_shape[1],
#         #               out_features=RoIPooling_shape[0] * RoIPooling_shape[1]),
#         #     nn.Linear(in_features=RoIPooling_shape[0] * RoIPooling_shape[1], out_features=n_classes)
#         # )
#           self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         feat_head = self.head(x)
#         feat_rdb = self.RDB(feat_head)
#         feat_conv = self.end(feat_rdb)
#         b, c, w, h = x.shape
#         out = self.MaxPool2d(feat_conv)
#         out = out.reshape(b, -1)
#         # rois = torch.tensor([[0, 0, 0, w, h] for x in range(b)]).cuda().double()
#         # feat_roipooling = ops.roi_pool(feat_conv, rois, self.RoIPooling_shape)  # x1, y1, x2, y2
#         # feat_roipooling = feat_roipooling.reshape(b, -1)
#         # out = self.Linear(feat_roipooling)
#         return self.softmax(out)

# class Classifier(nn.Module):
#     def __init__(self, n_classes=3, in_channels=3, RoIPooling_shape: list = [7, 7], img_shape=[3, 128, 128],
#                  channels: int = 64, grow_channels: int = 32, scale_ratio: float = 0.2):
#         super(Classifier, self).__init__()
#
#         self.n_classes = n_classes
#
#         def discriminator_block(in_filters, out_filters, normalization=True):
#             """Returns downsampling layers of each discriminator block"""
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
#             if normalization:
#                 layers.append(nn.InstanceNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers
#
#         self.model = nn.Sequential(
#             *discriminator_block(in_channels, 64, normalization=False),
#             *discriminator_block(64, 128),
#             *discriminator_block(128, 256),
#             *discriminator_block(256, 512),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#         )
#
#         # self.adv_layer = nn.Sequential(nn.Conv2d(512, 1, 4, padding=1, bias=False))
#         self.latent_layer = nn.Sequential(
#             nn.Conv2d(512, n_classes, kernel_size=3, padding=1, bias=False))  # patch size
#
#         self.MaxPooling = nn.AdaptiveAvgPool2d((1, 1))
#
#         self.softmax = nn.Softmax(dim=1)
#
#     def forward(self, x):
#         # Concatenate image and condition image by channels to produce input
#         out = self.model(x)
#         latent_code = self.latent_layer(out)
#         latent_code_mean = self.MaxPooling(latent_code)
#         return latent_code_mean.view(x.shape[0], self.n_classes)


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
            nn.Conv2d(256, n_classes, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.MaxPooling = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.model(x)
        out = self.MaxPooling(out)
        out = out.view(out.shape[0], self.n_classes)
        return self.softmax(out)

class RRDB(nn.Module):
    r"""
    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, grow_channels: int = 32, scale_ratio: float = 0.2):
        super(RRDB, self).__init__()

        self.RDB1 = RDB(channels, grow_channels)
        self.RDB2 = RDB(channels, grow_channels)
        self.RDB2 = RDB(channels, grow_channels)

        self.scale_ratio = scale_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        out = torch.add(out * self.scale_ratio, x)

        return out
