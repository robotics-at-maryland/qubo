
#import torch
import torch.nn as nn
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

def Qestimator(**args):
    model_ft = models.resnet18(pretrained = True)
    num_ftrs = model_ft.in_features
    model_ft.fc = nn.Linear( num_ftrs, args['num_label'] )
