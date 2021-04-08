# source code
# https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On/tree/master/Chapter06

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[2], 16, kernel_size = 5, stride= 2)
    '''    
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    '''
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.head(x.view(x.size(0), -1))
    