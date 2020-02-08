import torch
import torch.nn as nn
from torchvision import models


def get_vgg_encoder(vgg_type, num_features):
    """
    :param vgg_type: classname of desired vgg model, e.g. torchvision.models.vgg16
    :param num_features: number of output features for encoder
    :return: vgg model (nn.Module type)
    """
    model = vgg_type(pretrained=False, progress=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_features)
    return model

