'''define the config file for cityscapes and resnet50os8'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'cityscapes',
    'rootdir': os.path.join(os.getcwd(), 'CityScapes'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': (0.5, 2.0)}),
    ('RandomCrop', {'crop_size': (512, 1024), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (512, 1024), 'data_type': 'tensor'}),
]
DATASET_CFG['train']['set'] = 'trainval'
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (2048, 1024), 'keep_ratio': True, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'max_epochs': 440,
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update(
    {
        'num_classes': 19,
        'backbone': {
            'type': 'resnet50',
            'series': 'resnet',
            'pretrained': True,
            'outstride': 8,
            'use_stem': True,
            'selected_indices': (0, 1, 2, 3),
        },
    }
)
SEGMENTOR_CFG['memory']['use_loss'] = False
SEGMENTOR_CFG['memory']['update_cfg']['momentum_cfg']['base_lr'] = 0.01 * 0.9
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'memorynet_resnet50os8_cityscapes_train',
        'logfilepath': 'memorynet_resnet50os8_cityscapes_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'memorynet_resnet50os8_cityscapes_test',
        'logfilepath': 'memorynet_resnet50os8_cityscapes_test/test.log',
        'resultsavepath': 'memorynet_resnet50os8_cityscapes_test/memorynet_resnet50os8_cityscapes_results.pkl'
    }
)