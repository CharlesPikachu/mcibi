'''
Function:
    Implementation of ISANet
Author:
    Zhenchao Jin
'''
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from ..base import SelfAttentionBlock as _SelfAttentionBlock
from ...backbones import BuildActivation, BuildNormalization


'''Self-Attention Module'''
class SelfAttentionBlock(_SelfAttentionBlock):
    def __init__(self, in_channels, feats_channels, norm_cfg, act_cfg, **kwargs):
        super(SelfAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            transform_channels=feats_channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=False,
            matmul_norm=True,
            with_out_project=False,
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg)
        )
        self.output_project = self.buildproject(
            in_channels=in_channels,
            out_channels=in_channels,
            num_convs=1,
            use_norm=True,
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg),
        )
    '''forward'''
    def forward(self, x):
        context = super(SelfAttentionBlock, self).forward(x, x)
        return self.output_project(context)


'''ISANet'''
class ISANet(BaseModel):
    def __init__(self, cfg, **kwargs):
        super(ISANet, self).__init__(cfg, **kwargs)
        align_corners, norm_cfg, act_cfg = self.align_corners, self.norm_cfg, self.act_cfg
        # build isa module
        isa_cfg = cfg['isa']
        self.down_factor = isa_cfg['down_factor']
        self.in_conv = nn.Sequential(
            nn.Conv2d(isa_cfg['in_channels'], isa_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (isa_cfg['feats_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        self.global_relation = SelfAttentionBlock(
            in_channels=isa_cfg['feats_channels'],
            feats_channels=isa_cfg['isa_channels'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg)
        )
        self.local_relation = SelfAttentionBlock(
            in_channels=isa_cfg['feats_channels'],
            feats_channels=isa_cfg['isa_channels'],
            norm_cfg=copy.deepcopy(norm_cfg),
            act_cfg=copy.deepcopy(act_cfg)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(isa_cfg['feats_channels'] * 2, isa_cfg['feats_channels'], kernel_size=3, stride=1, padding=1, bias=False),
            BuildNormalization(norm_cfg['type'], (isa_cfg['feats_channels'], norm_cfg['opts'])),
            BuildActivation(act_cfg['type'], **act_cfg['opts']),
        )
        # build decoder
        decoder_cfg = cfg['decoder']
        self.decoder = nn.Sequential(
            nn.Dropout2d(decoder_cfg['dropout']),
            nn.Conv2d(decoder_cfg['in_channels'], cfg['num_classes'], kernel_size=1, stride=1, padding=0)
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
        # feed to isa module
        feats = self.in_conv(backbone_outputs[-1])
        residual = feats
        n, c, h, w = feats.size()
        loc_h, loc_w = self.down_factor
        glb_h, glb_w = math.ceil(h / loc_h), math.ceil(w / loc_w)
        pad_h, pad_w = glb_h * loc_h - h, glb_w * loc_w - w
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            feats = F.pad(feats, padding)
        # --global relation
        feats = feats.view(n, c, glb_h, loc_h, glb_w, loc_w)
        # ----do permutation to gather global group
        feats = feats.permute(0, 3, 5, 1, 2, 4)
        feats = feats.reshape(-1, c, glb_h, glb_w)
        # ----apply attention within each global group
        feats = self.global_relation(feats)
        # --local relation
        feats = feats.view(n, loc_h, loc_w, c, glb_h, glb_w)
        # ----do permutation to gather local group
        feats = feats.permute(0, 4, 5, 3, 1, 2)
        feats = feats.reshape(-1, c, loc_h, loc_w)
        # ----apply attention within each local group
        feats = self.local_relation(feats)
        # --permute each pixel back to its original position
        feats = feats.view(n, glb_h, glb_w, c, loc_h, loc_w)
        feats = feats.permute(0, 3, 1, 4, 2, 5)
        feats = feats.reshape(n, c, glb_h * loc_h, glb_w * loc_w)
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h//2: pad_h//2+h, pad_w//2: pad_w//2+w]
        feats = self.out_conv(torch.cat([feats, residual], dim=1))
        # feed to decoder
        predictions = self.decoder(feats)
        # return according to the mode
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
            'in_conv': self.in_conv,
            'global_relation': self.global_relation,
            'local_relation': self.local_relation,
            'out_conv': self.out_conv,
            'decoder': self.decoder,
        }
        if hasattr(self, 'auxiliary_decoder'):
            all_layers['auxiliary_decoder'] = self.auxiliary_decoder
        return all_layers