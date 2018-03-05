#
import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
# Ultra-sound nerve segmentation

class Dice_Coef_Loss(nn.Module):
    def __init__(self, smooth=1.):
        super(Dice_Coef_Loss, self).__init__()
        self.smooth = smooth

    def dice_coef(self, y_pred, y_true ):
        batch_size = y_true.size()[0]
        y_true_f = y_true.view(batch_size, -1)
        y_pred_f = y_pred.view(batch_size, -1)
        intersection = torch.sum(y_true_f * y_pred_f, dim=1)
        #print(intersection.size())
        losses = (2. * intersection + self.smooth) / (torch.sum(y_true_f, dim=1) + torch.sum(y_pred_f, dim=1) + self.smooth)
        #print(losses.data)
        #print(losses.size())
        return torch.mean(losses, dim=0)

    def forward(self, x, label):
        return -1 * self.dice_coef(x, label)
