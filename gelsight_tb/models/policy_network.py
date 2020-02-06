import torch
import torch.nn as nn
import numpy as np
from gelsight_tb.models.modules.vgg_encoder import get_vgg_16_encoder


class PolicyNetwork(nn.Module):

    def __init__(self, conf):
        super(PolicyNetwork, self).__init__()
        self.conf = conf
        self.batch_size = self.conf.batch_size
        self.build_network()

    def build_network(self):
        num_image_inputs = self.conf.num_image_inputs
        self.image_encoders = [get_vgg_16_encoder(self.conf.encoder_features)]
        current_layer_width = len(self.image_encoders * self.conf.encoder_features)
        self.fc_layers = []
        for layer in self.conf.policy_layers:
            self.fc_layers.append(nn.Linear(current_layer_width, layer['num_features']))
            current_layer_width = layer['num_features']
        self.output_layer = nn.Linear(current_layer_width, self.conf.action_dim)


