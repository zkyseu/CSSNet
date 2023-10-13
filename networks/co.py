import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SepConv2d(nn.Module):
    """
    深度可分离卷积
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm2d(in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

    
class conv_pos(nn.Module):
    def __init__(self,dim):
        super(conv_pos,self).__init__()
    
        self.conv1 = nn.Conv2d(dim,dim,1)
        self.conv2 = nn.Conv2d(dim,dim,1)
        self.sep = SepConv2d(dim,dim,3,padding = 1)

    
    def forward(self,x,size):
        B,C,H,W = x.size()
        N = H*W
        H = W = size

        x1 = self.conv1(x)
        x2 = self.conv2(x).reshape(B,C,N)

        x1 = self.sep(x1)
        x1 = x1.reshape(B,C,N)
        out = x1*x2
        return out