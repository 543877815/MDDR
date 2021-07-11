from torchvision.models import vgg19
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, feature_layer=35):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:feature_layer])

    def forward(self, img):
        return self.vgg19_54(img)
