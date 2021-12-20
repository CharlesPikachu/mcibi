'''
Function:
    Implementation of SemanticFPN
Author:
    Zhenchao Jin
'''
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from ..base import FPN, BaseModel
from ...backbones import BuildActivation, BuildNormalization


'''SemanticFPN'''
class SemanticFPN(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(SemanticFPN, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build fpn
        fpn_cfg = cfg['fpn']
        self.fpn_neck = FPN(
            in_channels_list=fpn_cfg['in_channels_list'],
            out_channels=fpn_cfg['out_channels'],
            upsample_cfg=fpn_cfg['upsample_cfg'],
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )
        self.scale_heads, feature_stride_list = nn.ModuleList(), fpn_cfg['feature_stride_list']
        for i in range(len(feature_stride_list)):
            head_length = max(1, int(np.log2(feature_stride_list[i]) - np.log2(feature_stride_list[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(nn.Sequential(
                    nn.Conv2d(fpn_cfg['out_channels'] if k == 0 else fpn_cfg['scale_head_channels'], fpn_cfg['scale_head_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                    BuildNormalization(norm_cfg['type'], (fpn_cfg['scale_head_channels'], norm_cfg['opts'])),
                    BuildActivation(act_cfg['type'], **act_cfg['opts']),
                ))
                if feature_stride_list[i] != feature_stride_list[0]:
                    scale_head.append(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=align_corners)
                    )
            self.scale_heads.append(nn.Sequential(*scale_head))
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to fpn
        fpn_outs = self.fpn_neck(list(backbone_outputs))
        feats = self.scale_heads[0](fpn_outs[0])
        for i in range(1, len(self.cfg['fpn']['feature_stride_list'])):
            feats = feats + F.interpolate(self.scale_heads[i](fpn_outs[i]), size=feats.shape[2:], mode='bilinear', align_corners=self.align_corners)
        # feed to decoder
        predictions = self.decoder(feats)
        # forward according to the mode
        if self.mode == 'TRAIN':
            loss, losses_log_dict = self.forwardtrain(
                predictions=predictions,
                targets=targets,
                backbone_outputs=backbone_outputs,
                losses_cfg=losses_cfg,
                img_size=img_size,
            )
            return loss, losses_log_dict
        return predictions
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'fpn_neck': self.fpn_neck,
            'scale_heads': self.scale_heads,
            'decoder': self.decoder,
        }