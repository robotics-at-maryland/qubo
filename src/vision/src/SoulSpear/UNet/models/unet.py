import torch
import torch.nn as nn
import torch.nn.functional as F
from .loss.dice import Dice_Coef_Loss
from .layers import bn_ccm, bn_cc, Upsample

__all__ = ['UNet']


class UNet(nn.Module):
    def __init__(self, num_mask, width, height):
        super(UNet, self).__init__()
        self.width = width
        self.height = height
        self.in_channel = 3
        self.start_channel = 16
        self.erudite = 0
        self.num_mask = num_mask
        sc = self.start_channel
        self.conv1 = bn_cc(self.in_channel, sc, 3, 1, 1 )
        self.conv2 = bn_cc(sc*1, sc*2, 3, 1, 1)
        self.conv3 = bn_cc(sc*2, sc*4, 3, 1, 1)
        self.conv4 = bn_cc(sc*4, sc*8, 3, 1, 1)
        self.conv5 = bn_cc(sc*8, sc*16, 3, 1, 1)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_conv1 = Upsample(sc*16, sc*8, sc*8, 2, 2, 0, 0)
        self.up_conv2 = Upsample(sc*8, sc*4, sc*4, 2, 2, 0, 0)
        self.up_conv3 = Upsample(sc*4, sc*2, sc*2, 2, 2, 0, 0)
        self.up_conv4 = Upsample(sc*2, sc*1, sc*1, 2, 2, 0, 0)

        if num_mask == 1:
            self.out_conv = nn.Sequential(
                nn.Conv2d(sc*1, num_mask, kernel_size=1),
                nn.Sigmoid(),
                )

        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(sc*1, num_mask, kernel_size=1),
                nn.LogSoftmax(dim=1)
                )
        self.loss = Dice_Coef_Loss()

    def forward(self, x):
        #print(x.size())
        x1 = self.conv1(x)
        #print(x1.size())
        x2 = self.downsample(x1)
        x2 = self.conv2(x2)
        #print(x2.size())
        x3 = self.downsample(x2)
        x3 = self.conv3(x3)
        #print(x3.size())
        x4 = self.downsample(x3)
        x4 = self.conv4(x4)
        #print(x4.size())
        x5 = self.downsample(x4)
        ux = self.conv5(x5)
        #print(ux.size())
        ux = self.up_conv1(x4, ux)
        #print(ux.size())
        ux = self.up_conv2(x3, ux)
        #print(ux.size())
        ux = self.up_conv3(x2, ux)
        #print(ux.size())
        ux = self.up_conv4(x1, ux)
        #print(ux.size())
        return self.out_conv(ux)
