'''
Function:
    A learnable scale parameter
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn


'''A learnable scale parameter'''
class Scale(nn.Module):
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
    '''forward'''
    def forward(self, x):
        return x * self.scale