'''
Function:
    Implementation of SETR
Author:
    Zhenchao Jin
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mla import MLANeck
from ..base import BaseModel
from ...backbones import BuildActivation, BuildNormalization


'''Naive upsampling head and Progressive upsampling head of SETR'''
class SETRUP(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(SETRUP, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build norm layer
        self.norm_layers = nn.ModuleList()
        for in_channels in cfg['normlayer']['in_channels_list']:
            norm_layer = BuildNormalization(cfg['normlayer']['type'], (in_channels, cfg['normlayer']['opts']))
            self.norm_layers.append(norm_layer)
        # build decoder
        self.decoder = self.builddecoder(cfg['decoder'])
        # build auxiliary decoder
        auxiliary_cfg_list = cfg['auxiliary']
        assert isinstance(auxiliary_cfg_list, (tuple, list))
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to norm layer
        assert len(backbone_outputs) == len(self.norm_layers)
        for idx in range(len(backbone_outputs)):
            backbone_outputs[idx] = self.norm(backbone_outputs[idx], self.norm_layers[idx])
        # feed to decoder
        predictions = self.decoder(backbone_outputs[-1])
        # forward according to the mode
        if self.mode == 'TRAIN':
            predictions = F.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)
            outputs_dict = {'loss_cls': predictions}
            backbone_outputs = backbone_outputs[:-1]
            for idx, (out, dec) in enumerate(zip(backbone_outputs, self.auxiliary_decoders)):
                predictions_aux = dec(out)
                predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                outputs_dict[f'loss_aux{idx+1}'] = predictions_aux
            return self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return predictions
    '''norm layer'''
    def norm(self, x, norm_layer):
        n, c, h, w = x.shape
        x = x.reshape(n, c, h * w).transpose(2, 1).contiguous()
        x = norm_layer(x)
        x = x.transpose(1, 2).reshape(n, c, h, w).contiguous()
        return x
    '''build decoder'''
    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])))
            layers.append(BuildActivation(act_cfg['type'], **act_cfg['opts']))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'norm_layers': self.norm_layers,
            'decoder': self.decoder,
            'auxiliary_decoders': self.auxiliary_decoders
        }


'''Multi level feature aggretation head of SETR'''
class SETRMLA(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(SETRMLA, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build mla neck
        norm_layers = nn.ModuleList()
        for in_channels in cfg['normlayer']['in_channels_list']:
            norm_layer = BuildNormalization(cfg['normlayer']['type'], (in_channels, cfg['normlayer']['opts']))
            norm_layers.append(norm_layer)
        mla_cfg = cfg['mla']
        self.mla_neck = MLANeck(
            in_channels_list=mla_cfg['in_channels_list'], 
            out_channels=mla_cfg['out_channels'], 
            norm_layers=norm_layers, 
            norm_cfg=norm_cfg, 
            act_cfg=act_cfg,
        )
        # build upsample convs and decoder
        decoder_cfg = cfg['decoder']
        assert decoder_cfg['mla_channels'] * len(decoder_cfg['in_channels_list']) == decoder_cfg['out_channels']
        self.up_convs = nn.ModuleList()
        for i in range(len(decoder_cfg['in_channels_list'])):
            self.up_convs.append(nn.Sequential(
                nn.Conv2d(decoder_cfg['in_channels_list'][i], decoder_cfg['mla_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(norm_cfg['type'], (decoder_cfg['mla_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
                nn.Conv2d(decoder_cfg['mla_channels'], decoder_cfg['mla_channels'], kernel_size=3, stride=1, padding=1, bias=False),
                BuildNormalization(norm_cfg['type'], (decoder_cfg['mla_channels'], norm_cfg['opts'])),
                BuildActivation(act_cfg['type'], **act_cfg['opts']),
                nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners)
            ))
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['out_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0),
        )
        # build auxiliary decoder
        auxiliary_cfg_list = cfg['auxiliary']
        assert isinstance(auxiliary_cfg_list, (tuple, list))
        self.auxiliary_decoders = nn.ModuleList()
        for auxiliary_cfg in auxiliary_cfg_list:
            decoder = self.builddecoder(auxiliary_cfg)
            self.auxiliary_decoders.append(decoder)
        # freeze normalization layer if necessary
        if cfg.get('is_freeze_norm', False): self.freezenormalization()
    '''forward'''
    def forward(self, x, targets=None, losses_cfg=None):
        img_size = x.size(2), x.size(3)
        # feed to backbone network
        backbone_outputs = self.transforminputs(self.backbone_net(x), selected_indices=self.cfg['backbone'].get('selected_indices'))
        # feed to mla neck
        feats_list = self.mla_neck(list(backbone_outputs))
        # feed to decoder
        outputs = []
        assert len(feats_list) == len(self.up_convs)
        for feats, up_conv in zip(feats_list, self.up_convs):
            outputs.append(up_conv(feats))
        outputs = torch.cat(outputs, dim=1)
        predictions = self.decoder(outputs)
        # forward according to the mode
        if self.mode == 'TRAIN':
            predictions = F.interpolate(predictions, size=img_size, mode='bilinear', align_corners=self.align_corners)
            outputs_dict = {'loss_cls': predictions}
            feats_list = feats_list[-len(self.auxiliary_decoders):]
            for idx, (out, dec) in enumerate(zip(feats_list, self.auxiliary_decoders)):
                predictions_aux = dec(out)
                predictions_aux = F.interpolate(predictions_aux, size=img_size, mode='bilinear', align_corners=self.align_corners)
                outputs_dict[f'loss_aux{idx+1}'] = predictions_aux
            return self.calculatelosses(
                predictions=outputs_dict, 
                targets=targets, 
                losses_cfg=losses_cfg
            )
        return predictions
    '''build decoder'''
    def builddecoder(self, decoder_cfg):
        layers, norm_cfg, act_cfg, num_classes, align_corners, kernel_size = [], self.norm_cfg.copy(), self.act_cfg.copy(), self.cfg['num_classes'], self.align_corners, decoder_cfg['kernel_size']
        for idx in range(decoder_cfg['num_convs']):
            if idx == 0:
                layers.append(nn.Conv2d(decoder_cfg['in_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            else:
                layers.append(nn.Conv2d(decoder_cfg['out_channels'], decoder_cfg['out_channels'], kernel_size=kernel_size, stride=1, padding=int(kernel_size - 1) // 2, bias=False))
            layers.append(BuildNormalization(norm_cfg['type'], (decoder_cfg['out_channels'], norm_cfg['opts'])))
            layers.append(BuildActivation(act_cfg['type'], **act_cfg['opts']))
            layers.append(nn.Upsample(scale_factor=decoder_cfg['scale_factor'], mode='bilinear', align_corners=align_corners))
        layers.append(nn.Dropout2d(decoder_cfg['dropout']))
        layers.append(nn.Conv2d(decoder_cfg['out_channels'], num_classes, kernel_size=1, stride=1, padding=0))
        return nn.Sequential(*layers)
    '''return all layers'''
    def alllayers(self):
        return {
            'backbone_net': self.backbone_net,
            'mla_neck': self.mla_neck,
            'up_convs': self.up_convs,
            'decoder': self.decoder,
            'auxiliary_decoders': self.auxiliary_decoders
        }