'''
Function:
    Implementation of ICNet
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
from .icneck import ICNeck
from ..base import BaseModel
from .icnetencoder import ICNetEncoder
from ...backbones import BuildActivation, BuildNormalization


'''ICNet'''
class ICNet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(ICNet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build encoder
        delattr(self, 'backbone_net')
        encoder_cfg = cfg['encoder']
        encoder_cfg.update({'backbone_cfg': cfg['backbone']})
        if 'act_cfg' not in encoder_cfg: encoder_cfg.update({'act_cfg': act_cfg})
        if 'norm_cfg' not in encoder_cfg: encoder_cfg.update({'norm_cfg': norm_cfg})
        if 'align_corners' not in encoder_cfg: encoder_cfg.update({'align_corners': align_corners})
        self.backbone_net = ICNetEncoder(**encoder_cfg)
        # build neck
        neck_cfg = cfg['neck']
        if 'act_cfg' not in neck_cfg: neck_cfg.update({'act_cfg': act_cfg})
        if 'norm_cfg' not in neck_cfg: neck_cfg.update({'norm_cfg': norm_cfg})
        if 'align_corners' not in neck_cfg: neck_cfg.update({'align_corners': align_corners})
        self.neck = ICNeck(**neck_cfg)
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
        )
        # build auxiliary decoder
        self.setauxiliarydecoder(cfg['auxiliary'])
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to neck
        backbone_outputs = self.neck(backbone_outputs)
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
            'neck': self.neck,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers