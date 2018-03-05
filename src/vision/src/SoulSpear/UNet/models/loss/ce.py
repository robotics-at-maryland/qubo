#
import torch
import torch.nn as nn
import torch.nn.functional as F

class CE_Loss(nn.Module):
    def __init__(self):
        super(ROI_CE_Loss, self).__init__()
        self.NLLLoss = nn.NLLLoss()

    def forward(self, x, label):
        num_channel = x.size()[1]
        num_batch = x.size()[0]
        self.NLLLoss(y, label)
