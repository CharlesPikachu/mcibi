'''
Function:
    Implementation of Swin
Author:
    Zhenchao Jin
'''
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .bricks import PatchEmbed as PatchEmbedBase
from .bricks import PatchMerging as PatchMergingBase
from .bricks import BuildNormalization, BuildActivation, BuildDropout, FFN


'''model urls'''
model_urls = {
    'swin_tiny_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth',
    'swin_small_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth',
    'swin_base_patch4_window12_384': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth',
    'swin_base_patch4_window7_224': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth',
    'swin_base_patch4_window12_384_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth',
    'swin_base_patch4_window7_224_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth',
    'swin_large_patch4_window12_384_22k': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
}


'''merge patch feature map'''
class PatchMerging(PatchMergingBase):
    def __init__(self, **kwargs):
        super(PatchMerging, self).__init__(**kwargs)
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        if self.norm is None: return {}
        return {'PatchMerging.norm': self.norm}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return {'PatchMerging.reduction': self.reduction}


'''Image to Patch Embedding'''
class PatchEmbed(PatchEmbedBase):
    def __init__(self, **kwargs):
        super(PatchEmbed, self).__init__(**kwargs)
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        if self.norm is None: return {}
        return {'PatchEmbed.norm': self.norm}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return {'PatchEmbed.projection': self.projection}


'''Window based multi-head self-attention (W-MSA) module with relative position bias'''
class WindowMSA(nn.Module):
    def __init__(self, embed_dims, num_heads, window_size, qkv_bias=True, qk_scale=None, attn_drop_rate=0., proj_drop_rate=0.):
        super(WindowMSA, self).__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        Wh, Ww = self.window_size
        rel_index_coords = self.doublestepseq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)
        self.softmax = nn.Softmax(dim=-1)
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        return {'WindowMSA.relative_position_bias_table': self.relative_position_bias_table}
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return {'WindowMSA.qkv': self.qkv, 'WindowMSA.proj': self.proj}
    '''forward'''
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    '''double step seq'''
    @staticmethod
    def doublestepseq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


'''Shift Window Multihead Self-Attention Module'''
class ShiftWindowMSA(nn.Module):
    def __init__(self, embed_dims, num_heads, window_size, shift_size=0, qkv_bias=True, qk_scale=None, attn_drop_rate=0, proj_drop_rate=0, dropout_cfg=None):
        super(ShiftWindowMSA, self).__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size
        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=(window_size, window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
        )
        self.drop = BuildDropout(dropout_cfg['type'], **dropout_cfg['opts'])
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        return self.w_msa.zerowdlayers()
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        return self.w_msa.nonzerowdlayers()
    '''forward'''
    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)
        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]
        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(query, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # nW, window_size, window_size, 1
            mask_windows = self.windowpartition(img_mask)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None
        # nW*B, window_size, window_size, C
        query_windows = self.windowpartition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)
        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        # B H' W' C
        shifted_x = self.windowreverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = self.drop(x)
        return x
    '''window reverse'''
    def windowreverse(self, windows, H, W):
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x
    '''window partition'''
    def windowpartition(self, x):
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


'''swin block'''
class SwinBlock(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, window_size=7, shift=False, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., act_cfg=None, norm_cfg=None):
        super(SwinBlock, self).__init__()
        self.norm1 = BuildNormalization(norm_cfg['type'], (embed_dims, norm_cfg['opts']))
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop_rate=attn_drop_rate, proj_drop_rate=drop_rate,
            dropout_cfg={'type': 'droppath', 'opts': {'drop_prob': drop_path_rate}},
        )
        self.norm2 = BuildNormalization(norm_cfg['type'], (embed_dims, norm_cfg['opts']))
        self.ffn = FFN(
            embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=2,
            ffn_drop=drop_rate, dropout_cfg={'type': 'droppath', 'opts': {'drop_prob': drop_path_rate}},
            act_cfg=act_cfg, add_identity=True,
        )
    '''layers with zero weight decay'''
    def zerowdlayers(self, apply_wd_to_relative_position_bias_table=True):
        # Note: our experiments find that do not set the weight decay of relative_position_bias_table as zero can achieve better segmentation performance
        if apply_wd_to_relative_position_bias_table: return {'SwinBlock.norm1': self.norm1, 'SwinBlock.norm2': self.norm2}
        layers = {'SwinBlock.norm1': self.norm1, 'SwinBlock.norm2': self.norm2}
        for key, value in self.attn.zerowdlayers().items():
            if 'relative_position_bias_table' in key:
                layers[f'SwinBlock.{key}'] = nn.ParameterList([value])
            else:
                layers[f'SwinBlock.{key}'] = value
        return layers
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self, apply_wd_to_relative_position_bias_table=True):
        # Note: our experiments find that do not set the weight decay of relative_position_bias_table as zero can achieve better segmentation performance
        if apply_wd_to_relative_position_bias_table: return {'SwinBlock.attn': self.attn, 'SwinBlock.ffn': self.ffn}
        layers = {'SwinBlock.ffn': self.ffn}
        for key, value in self.attn.nonzerowdlayers().items():
            if 'relative_position_bias_table' in key:
                layers[f'SwinBlock.{key}'] = nn.ParameterList([value])
            else:
                layers[f'SwinBlock.{key}'] = value
        return layers
    '''forward'''
    def forward(self, x, hw_shape):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = x + identity
        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x


'''Implements one stage in Swin Transformer'''
class SwinBlockSequence(nn.Module):
    def __init__(self, embed_dims, num_heads, feedforward_channels, depth, window_size=7, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., downsample=None, act_cfg=None, norm_cfg=None):
        super(SwinBlockSequence, self).__init__()
        drop_path_rate = drop_path_rate if isinstance(drop_path_rate, list) else [copy.deepcopy(drop_path_rate) for _ in range(depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims, num_heads=num_heads, feedforward_channels=feedforward_channels, window_size=window_size,
                shift=False if i % 2 == 0 else True, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate[i], act_cfg=act_cfg, norm_cfg=norm_cfg,
            )
            self.blocks.append(block)
        self.downsample = downsample
    '''layers with zero weight decay'''
    def zerowdlayers(self):
        layers = {}
        for layer_idx, block in enumerate(self.blocks):
            tmp_layers = block.zerowdlayers()
            new_tmp_layers = {}
            for key, value in tmp_layers.items():
                new_tmp_layers[f"{layer_idx}_{key}"] = value
            layers.update(new_tmp_layers)
        if self.downsample is not None:
            layers.update(self.downsample.zerowdlayers())
        return layers
    '''layers with non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {}
        for layer_idx, block in enumerate(self.blocks):
            tmp_layers = block.nonzerowdlayers()
            new_tmp_layers = {}
            for key, value in tmp_layers.items():
                new_tmp_layers[f"{layer_idx}_{key}"] = value
            layers.update(new_tmp_layers)
        if self.downsample is not None:
            layers.update(self.downsample.nonzerowdlayers())
        return layers
    '''forward'''
    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)
        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


'''Swin Transformer'''
class SwinTransformer(nn.Module):
    def __init__(self, pretrain_img_size=224, in_channels=3, embed_dims=96, patch_size=4, window_size=7, mlp_ratio=4, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2), out_indices=(0, 1, 2, 3), qkv_bias=True, qk_scale=None, patch_norm=True, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 use_abs_pos_embed=False, act_cfg=None, norm_cfg=None, **kwargs):
        super(SwinTransformer, self).__init__()
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = (pretrain_img_size, pretrain_img_size)
        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed
        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            kernel_size=patch_size,
            stride=strides[0],
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
        )
        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(torch.zeros((1, num_patches, embed_dims)))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        # stochastic depth
        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        self.stages = nn.ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                )
            else:
                downsample = None
            stage = SwinBlockSequence(
                embed_dims=in_channels, num_heads=num_heads[i], feedforward_channels=mlp_ratio * in_channels, depth=depths[i],
                window_size=window_size, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]): sum(depths[:i+1])], downsample=downsample, act_cfg=act_cfg, norm_cfg=norm_cfg
            )
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels
        self.num_features = [int(embed_dims * 2**i) for i in range(num_layers)]
        # add a norm layer for each output
        for i in out_indices:
            layer = BuildNormalization(norm_cfg['type'], (self.num_features[i], norm_cfg['opts']))
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)
    '''initialize backbone'''
    def initweights(self, swin_type='swin_tiny_patch4_window7_224', pretrained_model_path=''):
        # load
        if pretrained_model_path:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        else:
            checkpoint = model_zoo.load_url(model_urls[swin_type], map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        # be consistent
        state_dict = self.swinconvert(state_dict)
        state_dict_new = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                state_dict_new[k[9:]] = v
            else:
                state_dict_new[k] = v
        state_dict = state_dict_new
        # strip prefix of state_dict
        if list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        # reshape absolute position embedding
        if state_dict.get('absolute_pos_embed') is not None:
            absolute_pos_embed = state_dict['absolute_pos_embed']
            N1, L, C1 = absolute_pos_embed.size()
            N2, C2, H, W = self.absolute_pos_embed.size()
            if not (N1 != N2 or C1 != C2 or L != H * W):
                state_dict['absolute_pos_embed'] = absolute_pos_embed.view(N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
        # interpolate position bias table if needed
        relative_position_bias_table_keys = [k for k in state_dict.keys() if 'relative_position_bias_table' in k]
        for table_key in relative_position_bias_table_keys:
            table_pretrained = state_dict[table_key]
            table_current = self.state_dict()[table_key]
            L1, nH1 = table_pretrained.size()
            L2, nH2 = table_current.size()
            if not (nH1 != nH2):
                if L1 != L2:
                    S1 = int(L1**0.5)
                    S2 = int(L2**0.5)
                    table_pretrained_resized = F.interpolate(table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                    state_dict[table_key] = table_pretrained_resized.view(nH2, L2).permute(1, 0).contiguous()
        # load state_dict
        self.load_state_dict(state_dict, False)
    '''swin convert'''
    @staticmethod
    def swinconvert(ckpt):
        new_ckpt = OrderedDict()
        def correctunfoldreductionorder(x):
            out_channel, in_channel = x.shape
            x = x.reshape(out_channel, 4, in_channel // 4)
            x = x[:, [0, 2, 1, 3], :].transpose(1, 2).reshape(out_channel, in_channel)
            return x
        def correctunfoldnormorder(x):
            in_channel = x.shape[0]
            x = x.reshape(4, in_channel // 4)
            x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
            return x
        for k, v in ckpt.items():
            if k.startswith('head'):
                continue
            elif k.startswith('layers'):
                new_v = v
                if 'attn.' in k:
                    new_k = k.replace('attn.', 'attn.w_msa.')
                elif 'mlp.' in k:
                    if 'mlp.fc1.' in k: new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
                    elif 'mlp.fc2.' in k: new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
                    else: new_k = k.replace('mlp.', 'ffn.')
                elif 'downsample' in k:
                    new_k = k
                    if 'reduction.' in k: new_v = correctunfoldreductionorder(v)
                    elif 'norm.' in k: new_v = correctunfoldnormorder(v)
                else:
                    new_k = k
                new_k = new_k.replace('layers', 'stages', 1)
            elif k.startswith('patch_embed'):
                new_v = v
                if 'proj' in k:
                    new_k = k.replace('proj', 'projection')
                else:
                    new_k = k
            else:
                new_v = v
                new_k = k
            new_ckpt[new_k] = new_v
        return new_ckpt
    '''layers zero weight decay'''
    def zerowdlayers(self):
        layers = {}
        if hasattr(self, 'absolute_pos_embed'):
            layers['absolute_pos_embed'] = self.absolute_pos_embed
        layers.update(self.patch_embed.zerowdlayers())
        for layer_idx, stage in enumerate(self.stages):
            tmp_layers = stage.zerowdlayers()
            new_tmp_layers = {}
            for key, value in tmp_layers.items():
                new_tmp_layers[f"{layer_idx}_{key}"] = value
            layers.update(new_tmp_layers)
        for idx in self.out_indices:
            layers[f'SwinTransformer.norm{idx}'] = getattr(self, f'norm{idx}')
        return layers
    '''layers non zero weight decay'''
    def nonzerowdlayers(self):
        layers = {}
        layers.update(self.patch_embed.nonzerowdlayers())
        for layer_idx, stage in enumerate(self.stages):
            tmp_layers = stage.nonzerowdlayers()
            new_tmp_layers = {}
            for key, value in tmp_layers.items():
                new_tmp_layers[f"{layer_idx}_{key}"] = value
            layers.update(new_tmp_layers)
        return layers
    '''forward'''
    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs


'''build swin transformer'''
def BuildSwinTransformer(swin_type='swin_tiny_patch4_window7_224', **kwargs):
    # assert whether support
    supported_swins = {
        'swin_tiny_patch4_window7_224': {
            'pretrain_img_size': 224,
            'in_channels': 3,
            'embed_dims': 96,
            'patch_size': 4,
            'window_size': 7,
            'mlp_ratio': 4,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'qkv_bias': True,
            'qk_scale': None,
            'patch_norm': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.3,
            'use_abs_pos_embed': False,
        },
        'swin_small_patch4_window7_224': {
            'pretrain_img_size': 224,
            'in_channels': 3,
            'embed_dims': 96,
            'patch_size': 4,
            'window_size': 7,
            'mlp_ratio': 4,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'qkv_bias': True,
            'qk_scale': None,
            'patch_norm': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.3,
            'use_abs_pos_embed': False,
        },
        'swin_base_patch4_window12_384': {
            'pretrain_img_size': 384,
            'in_channels': 3,
            'embed_dims': 128,
            'patch_size': 4,
            'window_size': 12,
            'mlp_ratio': 4,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'qkv_bias': True,
            'qk_scale': None,
            'patch_norm': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.3,
            'use_abs_pos_embed': False,
        },
        'swin_base_patch4_window7_224': {
            'pretrain_img_size': 224,
            'in_channels': 3,
            'embed_dims': 128,
            'patch_size': 4,
            'window_size': 7,
            'mlp_ratio': 4,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'qkv_bias': True,
            'qk_scale': None,
            'patch_norm': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.3,
            'use_abs_pos_embed': False,
        },
        'swin_base_patch4_window12_384_22k': {
            'pretrain_img_size': 384,
            'in_channels': 3,
            'embed_dims': 128,
            'patch_size': 4,
            'window_size': 12,
            'mlp_ratio': 4,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'qkv_bias': True,
            'qk_scale': None,
            'patch_norm': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.3,
            'use_abs_pos_embed': False,
        },
        'swin_base_patch4_window7_224_22k': {
            'pretrain_img_size': 224,
            'in_channels': 3,
            'embed_dims': 128,
            'patch_size': 4,
            'window_size': 7,
            'mlp_ratio': 4,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'qkv_bias': True,
            'qk_scale': None,
            'patch_norm': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.3,
            'use_abs_pos_embed': False,
        },
        'swin_large_patch4_window12_384_22k': {
            'pretrain_img_size': 384,
            'in_channels': 3,
            'embed_dims': 192,
            'patch_size': 4,
            'window_size': 12,
            'mlp_ratio': 4,
            'depths': [2, 2, 18, 2],
            'num_heads': [6, 12, 24, 48],
            'qkv_bias': True,
            'qk_scale': None,
            'patch_norm': True,
            'drop_rate': 0.,
            'attn_drop_rate': 0.,
            'drop_path_rate': 0.3,
            'use_abs_pos_embed': False,
        },
    }
    assert swin_type in supported_swins, 'unsupport the swin_type %s...' % swin_type
    # parse args
    default_args = {
        'strides': (4, 2, 2, 2),
        'out_indices': (0, 1, 2, 3),
        'norm_cfg': {'type': 'layernorm', 'opts': {}},
        'act_cfg': {'type': 'gelu', 'opts': {}},
        'pretrained': True,
        'pretrained_model_path': '',
    }
    default_args.update(supported_swins[swin_type])
    for key, value in kwargs.items():
        if key in default_args: default_args.update({key: value})
    # obtain the instanced swin
    swin_args = default_args.copy()
    model = SwinTransformer(**swin_args)
    # load weights of pretrained model
    if default_args['pretrained']:
        model.initweights(swin_type, default_args['pretrained_model_path'])
    # return the model
    return model