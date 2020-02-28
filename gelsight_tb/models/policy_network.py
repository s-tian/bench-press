import torch
import torch.nn as nn
from torchvision import models
import numpy as np
from gelsight_tb.models.modules.pretrained_encoder import *
from gelsight_tb.models.model import Model


class PolicyNetwork(Model):

    activations = {
        'relu': torch.nn.ReLU(),
        'sigmoid': torch.nn.Sigmoid(),
        'leaky_relu': torch.nn.LeakyReLU()
    }

    input_sources = sorted(['external', 'gelsight_side', 'gelsight_top'])

    def __init__(self, conf, load_resume=None):
        super(PolicyNetwork, self).__init__(conf, load_resume)
        self.batch_size = self.conf.batch_size
        self.loss = nn.MSELoss()
        if self.conf.activation in self.activations:
            self.activation = self.activations[self.conf.activation]
        else:
            self.activation = lambda x: x

    def build_network(self):
        num_image_inputs = len(self.conf.image_inputs)
        if self.conf.encoder_type == 'resnet':
            self.image_encoders = nn.ModuleList(
                [get_resnet_encoder(models.resnet18, self.conf.encoder_features, freeze=False) for _ in range(num_image_inputs)])
        elif self.conf.encoder_type == 'resnet_spatial':
            self.image_encoders = nn.ModuleList(
                [get_resnet_spatial_encoder(models.resnet18, self.conf.encoder_features, freeze=False) for _ in range(num_image_inputs)])
        else:
            self.image_encoders = nn.ModuleList(
                [get_vgg_encoder(models.vgg13, self.conf.encoder_features) for _ in range(num_image_inputs)])

        if self.conf.use_state:
            current_layer_width = len(self.image_encoders) * self.conf.encoder_features + self.conf.state_dim
        else:
            print('Not using state! ')
            current_layer_width = len(self.image_encoders) * self.conf.encoder_features
        self.fc_layers = nn.ModuleList()
        for layer in self.conf.policy_layers:
            self.fc_layers.append(nn.Linear(current_layer_width, layer))
            current_layer_width = layer
        self.output_layer = nn.Linear(current_layer_width, self.conf.action_dim)

    def forward(self, inputs):
        """
        :param inputs: a dictionary containing the following keys:
            'images': a list of num_cameras image tensors of shape [B, C, W, H]
            'state': a tensor of shape [B, state_dim]
        :return: a tensor of shape [B, action_dim]
        """

        image_inputs, state_input = inputs['images'], inputs['state']
        sel_image_inputs = []
        for name in self.conf.image_inputs:
            assert name in self.input_sources
            ind = self.input_sources.index(name)
            sel_image_inputs.append(image_inputs[ind])

        image_inputs = sel_image_inputs
        image_encodings = []
        if self.image_encoders:
            for camera_i_images, encoder in zip(image_inputs, self.image_encoders):
                image_encodings.append(encoder(camera_i_images))
            image_encodings_cat = torch.cat(image_encodings, dim=1)  # form [B, num_cam*encoder_features] tensor
            output = image_encodings_cat
        if self.conf.use_state:
            if image_encodings: 
                output = torch.cat((image_encodings_cat, state_input), dim=1)
            else:
                output = state_input
        for layer in self.fc_layers:
            output = self.activation(layer(output))
        output = self.output_layer(output)
        return output

