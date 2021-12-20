'''
Function:
    Implementation of FCN
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from ..base import BaseModel
from ...backbones import BuildActivation, BuildNormalization, DepthwiseSeparableConv2d


'''FCN'''
class FCN(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(FCN, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build decoder
        decoder_cfg = cfg['decoder']
        convs = []
        for idx in range(decoder_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            else:
                conv = nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False)
            norm = BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts']))
            act = BuildActivation(act_cfg['type'], **act_cfg['opts'])
            convs += [conv, norm, act]
        convs.append(nn.Dropout2d(decoder_cfg['dropout']))
        if decoder_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to decoder
        predictions = self.decoder(backbone_outputs[-1])
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
            'backbone_net': self.backbone_net,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers


'''DepthwiseSeparableFCN'''
class DepthwiseSeparableFCN(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(DepthwiseSeparableFCN, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build decoder
        decoder_cfg = cfg['decoder']
        convs = []
        for idx in range(decoder_cfg.get('num_convs', 2)):
            if idx == 0:
                conv = DepthwiseSeparableConv2d(
                    in_channels=decoder_cfg['in_channels'],
                    out_channels=decoder_cfg['out_channels'],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            else:
                conv = DepthwiseSeparableConv2d(
                    in_channels=decoder_cfg['out_channels'],
                    out_channels=decoder_cfg['out_channels'],
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            convs.append(conv)
        convs.append(nn.Dropout2d(decoder_cfg['dropout']))
        if decoder_cfg.get('num_convs', 2) > 0:
            convs.append(nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        else:
            convs.append(nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0))
        self.decoder = nn.Sequential(*convs)
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to decoder
        predictions = self.decoder(backbone_outputs[-1])
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
            'backbone_net': self.backbone_net,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers