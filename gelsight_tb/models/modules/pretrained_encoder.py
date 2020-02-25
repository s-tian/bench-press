import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


pretrained_model_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])


def get_vgg_encoder(vgg_type, num_features):
    """
    :param vgg_type: classname of desired vgg model, e.g. torchvision.models.vgg16
    :param num_features: number of output features for encoder
    :return: vgg model (nn.Module type)
    """
    model = vgg_type(pretrained=True, progress=True)
    model.classifier[-1] = nn.Linear(in_features=4096, out_features=num_features)
    return model


def print_hook(self, input, output):
    print(f'output size: {output.data.size()}')
    print(f'output norm: {output.data.norm()}')


def get_resnet_encoder(resnet_type, num_features, freeze=False):
    model = resnet_type(pretrained=True, progress=True)
    for param in model.parameters():
        param.requires_grad = not freeze
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_features)
    avgpool = model.avgpool
    #avgpool.register_forward_hook(print_hook)
    #import ipdb; ipdb.set_trace()
    return model
