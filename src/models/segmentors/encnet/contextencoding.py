'''
Function:
    Implementation of Context Encoding Module
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoding import Encoding
from ...backbones import BuildActivation, BuildNormalization


'''Context Encoding Module'''
class ContextEncoding(nn.Module):
    def __init__(self, in_channels, num_codes, **kwargs):
        super(ContextEncoding, self).__init__()
        norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        self.encoding_project = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (in_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        encoding_norm_cfg = copy.deepcopy(norm_cfg)
        encoding_norm_cfg['type'] = encoding_norm_cfg['type'].replace('2d', '1d')
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            BuildNormalization(encoding_norm_cfg['type'], (num_codes, encoding_norm_cfg['opts'])),
            BuildActivation('relu', **{'inplace': True})
        )
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Sigmoid()
        )
    '''forward'''
    def forward(self, x):
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)
        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output