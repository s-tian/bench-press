import torch
import torch.nn as nn
from torchvision import models


def get_vgg_16_encoder(num_features):
    model = models.vgg16(pretrained=False)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_features)
    return model