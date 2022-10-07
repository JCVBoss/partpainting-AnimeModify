# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as tdist
import torchvision.models as models
import kornia
import imp

import numpy as np

# from .networks import Generator, MultiscaleDiscriminator, DenseD, Dis_Inn
from src.PPModule import Generator,MultiscaleDiscriminator,DenseD,Dis_Inn

import math


class PartPModel(nn.Module):
    def __init__(self):
        super(PartPModel, self).__init__()

        g = Generator()

        self.add_module('g', g)

        self.generator_path = ""

        # load pretrained_weight
        if os.path.exists(self.generator_path):
            print('Loading %s Model ...' % self.generator_path)

            g_data = torch.load(self.generator_path)
            self.g.load_state_dict(g_data['params'])
            self.iteration = g_data['iteration']


    def forward(self, pdata, half_fmask, pos=None, z=None):
        o, (mu, logvar), ys = self.g(pdata, 1 - half_fmask)
        return o, ys


