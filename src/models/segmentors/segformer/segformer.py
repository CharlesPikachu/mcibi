'''
Function:
    Implementation of Segformer
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from ...backbones import BuildActivation, BuildNormalization


'''Segformer'''
class Segformer(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(Segformer, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build decoder
        decoder_cfg = cfg['decoder']
        self.convs = nn.ModuleList()
        for in_channels in decoder_cfg['in_channels_list']:
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels, decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
                BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
            ))
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['out_channels'] * len(self.convs), decoder_cfg['out_channels'], kernel_size=1, stride=1, padding=0, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to decoder
        outs = []
        for idx, feats in enumerate(list(backbone_outputs)):
            outs.append(
                F.interpolate(self.convs[idx](feats), size=backbone_outputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            )
        feats = torch.cat(outs, dim=1)
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
        all_layers = {
            'convs': self.convs,
            'decoder': self.decoder,
        }
        tmp_layers = []
        for key, value in self.backbone_net.zerowdlayers().items():
            tmp_layers.append(value)
        all_layers.update({'backbone_net_zerowd': nn.Sequential(*tmp_layers)})
        tmp_layers = []
        for key, value in self.backbone_net.nonzerowdlayers().items():
            tmp_layers.append(value)
        all_layers.update({'backbone_net_nonzerowd': nn.Sequential(*tmp_layers)})
        return all_layers