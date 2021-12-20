'''
Function:
    Implementation of FastSCNN
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from .bricks import BuildNormalization, BuildActivation, DepthwiseSeparableConv2d, InvertedResidual


'''model urls'''
model_urls = {}


'''Pooling Pyramid Module used in PSPNet'''
class PoolingPyramidModule(nn.ModuleList):
    def __init__(self, pool_scales, in_channels, out_channels, norm_cfg, act_cfg, align_corners, **kwargs):
        super(PoolingPyramidModule, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        for pool_scale in pool_scales:
            self.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            ))
    '''forward'''
    def forward(self, x):
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                input=ppm_out,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs


'''Learning to downsample module'''
class LearningToDownsample(nn.Module):
    def __init__(self, in_channels, dw_channels, out_channels, norm_cfg=None, act_cfg=None, dw_act_cfg=None):
        super(LearningToDownsample, self).__init__()
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.dw_act_cfg = dw_act_cfg
        dw_channels1, dw_channels2 = dw_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, dw_channels1, kernel_size=3, stride=2, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (dw_channels1, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.dsconv1 = DepthwiseSeparableConv2d(
            in_channels=dw_channels1,
            out_channels=dw_channels2,
            kernel_size=3, 
            stride=2, 
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            dw_act_cfg=self.dw_act_cfg,
        )
        self.dsconv2 = DepthwiseSeparableConv2d(
            in_channels=dw_channels2,
            out_channels=out_channels,
            kernel_size=3, 
            stride=2, 
            padding=1,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            dw_act_cfg=self.dw_act_cfg,
        )
    '''forward'''
    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


'''Global feature extractor module'''
class GlobalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, expand_ratio=6, num_blocks=(3, 3, 3), strides=(2, 2, 1),
                 pool_scales=(1, 2, 3, 6), norm_cfg=None, act_cfg=None, align_corners=False):
        super(GlobalFeatureExtractor, self).__init__()
        # set attrs
        assert len(block_channels) == len(num_blocks) == 3
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        # define modules
        self.bottleneck1 = self.makelayer(in_channels, block_channels[0], num_blocks[0], strides[0], expand_ratio)
        self.bottleneck2 = self.makelayer(block_channels[0], block_channels[1], num_blocks[1], strides[1], expand_ratio)
        self.bottleneck3 = self.makelayer(block_channels[1], block_channels[2], num_blocks[2], strides[2], expand_ratio)
        self.ppm = PoolingPyramidModule(pool_scales, block_channels[2], block_channels[2] // 4, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, align_corners=align_corners)
        self.out = nn.Sequential(
            nn.Conv2d(block_channels[2] * 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
    '''make layer'''
    def makelayer(self, in_channels, out_channels, blocks, stride=1, expand_ratio=6):
        layers = [
            InvertedResidual(in_channels, out_channels, stride, expand_ratio, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        ]
        for i in range(1, blocks):
            layers.append(
                InvertedResidual(out_channels, out_channels, 1, expand_ratio, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
            )
        return nn.Sequential(*layers)
    '''forward'''
    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = torch.cat([x, *self.ppm(x)], dim=1)
        x = self.out(x)
        return x


'''Feature fusion module'''
class FeatureFusionModule(nn.Module):
    def __init__(self, higher_in_channels, lower_in_channels, out_channels, norm_cfg=None, dwconv_act_cfg=None, conv_act_cfg=None, align_corners=False):
        super(FeatureFusionModule, self).__init__()
        # set attrs
        self.norm_cfg = norm_cfg
        self.dwconv_act_cfg = dwconv_act_cfg
        self.conv_act_cfg = conv_act_cfg
        self.align_corners = align_corners
        # define modules
        self.dwconv = nn.Sequential(
            nn.Conv2d(lower_in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
            BuildActivation(dwconv_act_cfg['type'], **dwconv_act_cfg['opts']),
        )
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(higher_in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (out_channels, norm_cfg['opts'])),
        )
        self.act = BuildActivation(conv_act_cfg['type'], **conv_act_cfg['opts'])
    '''forward'''
    def forward(self, higher_res_feature, lower_res_feature):
        lower_res_feature = F.interpolate(lower_res_feature, size=higher_res_feature.size()[2:], mode='bilinear', align_corners=self.align_corners)
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.act(out)


'''FastSCNN'''
class FastSCNN(nn.Module):
    def __init__(self, in_channels=3, downsample_dw_channels=(32, 48), global_in_channels=64, global_block_channels=(64, 96, 128), global_block_strides=(2, 2, 1), global_out_channels=128, 
                 higher_in_channels=64, lower_in_channels=128, fusion_out_channels=128, out_indices=(0, 1, 2), norm_cfg=None, act_cfg=None, align_corners=False, dw_act_cfg=None, **kwargs):
        super(FastSCNN, self).__init__()
        assert global_in_channels == higher_in_channels, 'Global Input Channels must be the same with Higher Input Channels...'
        assert global_out_channels == lower_in_channels, 'Global Output Channels must be the same with Lower Input Channels...'
        # set attrs
        self.in_channels = in_channels
        self.downsample_dw_channels1 = downsample_dw_channels[0]
        self.downsample_dw_channels2 = downsample_dw_channels[1]
        self.global_in_channels = global_in_channels
        self.global_block_channels = global_block_channels
        self.global_block_strides = global_block_strides
        self.global_out_channels = global_out_channels
        self.higher_in_channels = higher_in_channels
        self.lower_in_channels = lower_in_channels
        self.fusion_out_channels = fusion_out_channels
        self.out_indices = out_indices
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.dw_act_cfg = dw_act_cfg
        # define modules
        self.learning_to_downsample = LearningToDownsample(
            in_channels=in_channels, 
            dw_channels=downsample_dw_channels, 
            out_channels=global_in_channels,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            dw_act_cfg=self.dw_act_cfg
        )
        self.global_feature_extractor = GlobalFeatureExtractor(
            in_channels=global_in_channels, 
            block_channels=global_block_channels, 
            out_channels=global_out_channels, 
            strides=self.global_block_strides,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
        )
        self.feature_fusion = FeatureFusionModule(
            higher_in_channels=higher_in_channels, 
            lower_in_channels=lower_in_channels, 
            out_channels=fusion_out_channels, 
            norm_cfg=self.norm_cfg, 
            dwconv_act_cfg=self.act_cfg, 
            conv_act_cfg=self.act_cfg, 
            align_corners=self.align_corners,
        )
    '''forward'''
    def forward(self, x):
        higher_res_features = self.learning_to_downsample(x)
        lower_res_features = self.global_feature_extractor(higher_res_features)
        fusion_output = self.feature_fusion(higher_res_features, lower_res_features)
        outs = [higher_res_features, lower_res_features, fusion_output]
        outs = [outs[i] for i in self.out_indices]
        return tuple(outs)


'''build fastscnn'''
def BuildFastSCNN(fastscnn_type=None, **kwargs):
    # assert whether support
    assert fastscnn_type is None
    # parse args
    default_args = {
        'in_channels': 3, 
        'downsample_dw_channels': (32, 48),
        'global_in_channels': 64,
        'global_block_channels': (64, 96, 128),
        'global_block_strides': (2, 2, 1),
        'global_out_channels': 128,
        'higher_in_channels': 64,
        'lower_in_channels': 128,
        'fusion_out_channels': 128,
        'out_indices': (0, 1, 2),
        'norm_cfg': None,
        'act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
        'align_corners': False,
        'dw_act_cfg': {'type': 'relu', 'opts': {'inplace': True}},
        'pretrained': False,
        'pretrained_model_path': '',
    }
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain args for instanced fastscnn
    fastscnn_args = default_args.copy()
    # obtain the instanced fastscnn
    model = FastSCNN(**fastscnn_args)
    # load weights of pretrained model
    if default_args['pretrained'] and os.path.exists(default_args['pretrained_model_path']):
        checkpoint = torch.load(default_args['pretrained_model_path'])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    elif default_args['pretrained']:
        checkpoint = model_zoo.load_url(model_urls[fastscnn_type])
        if 'state_dict' in checkpoint: state_dict = checkpoint['state_dict']
        else: state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
    # return the model
    return model