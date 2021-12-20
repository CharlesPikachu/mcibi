'''define the config file for lip and resnest101os8'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'lip',
    'rootdir': os.path.join(os.getcwd(), 'LIP'),
})
DATASET_CFG['train']['aug_opts'] = [
    ('Resize', {'output_size': (520, 520), 'keep_ratio': False, 'scale_range': (0.75, 1.25)}),
    ('RandomCrop', {'crop_size': (473, 473), 'one_category_max_ratio': 0.75}),
    ('RandomFlip', {'flip_prob': 0.5, 'fix_ann_pairs': [(15, 14), (17, 16), (19, 18)]}),
    ('RandomRotation', {'angle_upper': 30, 'rotation_prob': 0.6}),
    ('PhotoMetricDistortion', {}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
    ('Padding', {'output_size': (473, 473), 'data_type': 'tensor'}),
]
DATASET_CFG['test']['aug_opts'] = [
    ('Resize', {'output_size': (473, 473), 'keep_ratio': False, 'scale_range': None}),
    ('Normalize', {'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375]}),
    ('ToTensor', {}),
]
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
DATALOADER_CFG['train'].update(
    {
        'batch_size': 40,
    }
)
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
OPTIMIZER_CFG.update(
    {
        'type': 'sgd',
        'sgd': {
            'learning_rate': 0.007,
            'momentum': 0.9,
            'weight_decay': 5e-4,
        },
        'max_epochs': 150,
    }
)
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update(
    {
        'num_classes': 20,
        'backbone': {
            'type': 'resnest101',
            'series': 'resnest',
            'pretrained': True,
            'outstride': 8,
            'selected_indices': (0, 1, 2, 3),
        },
    }
)
SEGMENTOR_CFG['memory']['use_loss'] = False
SEGMENTOR_CFG['memory']['update_cfg']['momentum_cfg']['base_lr'] = 0.007
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['train'].update(
    {
        'backupdir': 'memorynet_resnest101os8_lip_train',
        'logfilepath': 'memorynet_resnest101os8_lip_train/train.log',
    }
)
COMMON_CFG['test'].update(
    {
        'backupdir': 'memorynet_resnest101os8_lip_test',
        'logfilepath': 'memorynet_resnest101os8_lip_test/test.log',
        'resultsavepath': 'memorynet_resnest101os8_lip_test/memorynet_resnest101os8_lip_results.pkl'
    }
)