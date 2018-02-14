import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxPoolStride1( nn.Module ):
    def __init__( self ):
        super( MaxPoolStride1, self).__init__()

    def forward( self, x ):
        # (pad_l, pad_r, pad_t, pad_b)
        return F.max_pool2d(F.pad(x, (0,1,0,1), mode='replicate'), 2, stride=1)


class Reorg(nn.Module):
    def __init__(self, stride=2):
        super(Reorg, self).__init__()
        self.stride = stride

    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view(B, C, H//hs, hs, W//ws, ws).transpose(3,4).contiguous()
        x = x.view(B, C, H//hs*W//ws, hs*ws).transpose(2,3).contiguous()
        x = x.view(B, C, hs*ws, H//hs, W//ws).transpose(1,2).contiguous()
        x = x.view(B, hs*ws*C, H//hs, W//ws)
        return x

class Route(nn.Module):
    def __init__( self, stride ):
        super(route, self).__init__()
        self.stride = stride
    def forward( *inputs ):
        return torch.cat( inputs, 1 )

class GlobalAvgPool2d(nn.Module):
    def __init__( self ):
        super( GlobalAvgPool2d, self).__init__()
    def forward( self , x):
        N, C, H, W  = x.size()
        return F.avg_pool2d( x, kernel_size = (H,W) ).view( N, C )

class Dunkey( nn.Module ):
    def __init__( self ):
        super(Dunkey,self).__init__()
    def forward( self , x ):
        return x

'''
class conv(nn.Module):
    def __init__(self, in_channel,out_channel, kernel_size, stride, padding):
        super(conv,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.layer= nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(negative_slope=0.1, inplace=False)
            )
    def forward(self,x):
        return self.layer(x)
'''

def conv( in_channel,out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channel),
        nn.LeakyReLU(negative_slope=0.1, inplace=False)
        )
