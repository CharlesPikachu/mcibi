'''memorynet_deeplabv3_resnet101os8_cocostuff'''
import os
from .base_cfg import *


# modify dataset config
DATASET_CFG = DATASET_CFG.copy()
DATASET_CFG.update({
    'type': 'cocostuff',
    'rootdir': os.path.join(os.getcwd(), 'COCO'),
})
# modify dataloader config
DATALOADER_CFG = DATALOADER_CFG.copy()
# modify optimizer config
OPTIMIZER_CFG = OPTIMIZER_CFG.copy()
# modify scheduler config
SCHEDULER_CFG = SCHEDULER_CFG.copy()
SCHEDULER_CFG.update({
    'max_epochs': 30
})
# modify losses config
LOSSES_CFG = LOSSES_CFG.copy()
# modify segmentor config
SEGMENTOR_CFG = SEGMENTOR_CFG.copy()
SEGMENTOR_CFG.update({
    'num_classes': 182,
})
# modify inference config
INFERENCE_CFG = INFERENCE_CFG.copy()
# modify common config
COMMON_CFG = COMMON_CFG.copy()
COMMON_CFG['work_dir'] = 'memorynet_deeplabv3_resnet101os8_cocostuff'
COMMON_CFG['logfilepath'] = 'memorynet_deeplabv3_resnet101os8_cocostuff/memorynet_deeplabv3_resnet101os8_cocostuff.log'
COMMON_CFG['resultsavepath'] = 'memorynet_deeplabv3_resnet101os8_cocostuff/memorynet_deeplabv3_resnet101os8_cocostuff_results.pkl'