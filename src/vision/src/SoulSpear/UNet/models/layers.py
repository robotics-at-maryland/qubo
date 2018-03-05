import torch
import torch.nn as nn
import torch.nn.functional as F

def bn_ccm(inc, oc, kernel_size, stride=None, padding=0 ):
    return nn.Sequential(
        nn.Conv2d(inc, oc, kernel_size, 1, padding),
        nn.ELU(inplace=True),
        #nn.Dropout2d(0.1),
        nn.BatchNorm2d(oc),
        nn.Conv2d(oc, oc, kernel_size, 1, padding),
        nn.ELU(inplace=True),
        #nn.Dropout2d(0.1),
        nn.BatchNorm2d(oc),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
def bn_cc(inc, oc, kernel_size, stride=None, padding=0 ):
    return nn.Sequential(
        nn.Conv2d(inc, oc, kernel_size, 1, padding),
        nn.ELU(inplace=True),
        #nn.Dropout2d(0.1),
        nn.BatchNorm2d(oc),
        nn.Conv2d(oc, oc, kernel_size, 1, padding),
        nn.ELU(inplace=True),
        #nn.Dropout2d(0.1),
        nn.BatchNorm2d(oc),
    )

class Upsample(nn.Module):
    def __init__(self, inc_u, inc_c, oc, kernel_size, stride, padding, output_padding ):
        super(Upsample,self).__init__()
        self.convT = nn.ConvTranspose2d( inc_u, oc,
            kernel_size, stride,
            padding, output_padding
            )
        self.unite = nn.Sequential(
            nn.Conv2d(oc + inc_c, oc, (3,3), padding=1),
            nn.ELU(inplace=True),
            nn.Dropout2d(0.2),
            nn.Conv2d(oc, oc, (3,3), padding=1),
            nn.ELU(inplace=True)
            )

    def forward(self, conv_x, up_x ):
        # Upsample the input from the previous layer and concatnate them with
        # forward feature map. The unite module make them less alienated.
        up_x = self.convT(up_x)
        #print(up_x.size())
        x = torch.cat([conv_x, up_x], dim=1)
        x = self.unite(x)
        return x
