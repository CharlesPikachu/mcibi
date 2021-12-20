'''
Function:
    Implementation of Dynamic Convolutional Module
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalization


'''Dynamic Convolutional Module'''
class DynamicConvolutionalModule(nn.Module):
    def __init__(self, filter_size, is_fusion, in_channels, out_channels, **kwargs):
        super(DynamicConvolutionalModule, self).__init__()
        norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        self.filter_size, self.is_fusion = filter_size, is_fusion
        self.filter_gen_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.input_redu_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.norm = BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts']))
        self.activate = BuildActivation(act_cfg['type'], **act_cfg['opts'])
        if is_fusion:
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            )
    '''forward'''
    def forward(self, x):
        generated_filter = self.filter_gen_conv(F.adaptive_avg_pool2d(x, self.filter_size))
        x = self.input_redu_conv(x)
        b, c, h, w = x.shape
        x = x.view(1, b * c, h, w)
        generated_filter = generated_filter.view(b * c, 1, self.filter_size, self.filter_size)
        pad = (self.filter_size - 1) // 2
        if (self.filter_size - 1) % 2 == 0:
            p2d = (pad, pad, pad, pad)
        else:
            p2d = (pad + 1, pad, pad + 1, pad)
        x = F.pad(input=x, pad=p2d, mode='constant', value=0)
        output = F.conv2d(input=x, weight=generated_filter, groups=b * c)
        output = output.view(b, c, h, w)
        output = self.norm(output)
        output = self.activate(output)
        if self.is_fusion: output = self.fusion_conv(output)
        return output