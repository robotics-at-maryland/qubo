#!/usr/bin/env python

#import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision
from torchvision import models

'''
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
'''
# Sourch Code
# https://github.com/pytorch/vision/blob/master/torchvision/models/

def Qestimator_resnet18(**args):
    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear( num_ftrs, args['num_label'] )
    return model_ft

'''
def Qestimator_vgg16(**args):
    model_ft = models.vgg16(pretrained = True )
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear( num_ftrs, args['num_label'] )
    return model_ft
'''
class Qestimator_naive(nn.Module):
    #Reference: http://pytorch.org/docs/master/nn.html
    def __init__(self, num_label, input_h):
        super(Qestimator_naive, self ).__init__()
        self.conv1 = nn.Conv2d(3 ,16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16 ,32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32 ,32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        Hout = (((input_h-4)/2 -4 )/2 -4)/2
        Flatten = 32*Hout**2 # assuming square input
        self.head = nn.Linear( Flatten, num_label ) # the 448 and 2

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head( x.view(x.size(0), -1) ) # flatten and feed it to output layer
