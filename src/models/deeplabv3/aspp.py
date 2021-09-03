'''
Function:
    define the Atrous Spatial Pyramid Pooling (ASPP)
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ...backbones import BuildActivation, BuildNormalization


'''ASPP'''
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations, **kwargs):
        super(ASPP, self).__init__()
        align_corners, norm_cfg, act_cfg = kwargs['align_corners'], kwargs['norm_cfg'], kwargs['act_cfg']
        self.align_corners = align_corners
        self.parallel_branches = nn.ModuleList()
        for idx, dilation in enumerate(dilations):
            if dilation == 1:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=dilation, bias=False),
                    BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
                    BuildActivation(act_cfg['type'], **act_cfg['opts'])
                )
            else:
                branch = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False),
                    BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
                    BuildActivation(act_cfg['type'], **act_cfg['opts'])
                )
            self.parallel_branches.append(branch)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * (len(dilations) + 1), out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts'])
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    '''forward'''
    def forward(self, x):
        size = x.size()
        outputs = []
        for branch in self.parallel_branches:
            outputs.append(branch(x))
        global_features = self.global_branch(x)
        global_features = F.interpolate(global_features, size=(size[2], size[3]), mode='bilinear', align_corners=self.align_corners)
        outputs.append(global_features)
        features = torch.cat(outputs, dim=1)
        features = self.bottleneck(features)
        return features