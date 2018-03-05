#
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
class ROILoss(nn.Module):
    def __init__(self, threshold):
        self.threshold = threshold

    def forward(self, x, label):
        '''
        Arguments:
         x: the predicted segementation of the image, continuous
         label: ground truth segementation (binary)
        '''
        x = x > self.threshold
        intersection = torch.sum( x*label > 0, dim = (1,2,3) )# element wise mul
        union = torch.sum( torch.(x label) > 0, dim= (1,2,3) )
        return intersection/union
"""
