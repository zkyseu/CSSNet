import torch 
import torch.nn as nn
import torch.nn.functional as F


class external_attention2(nn.Module):
    def __init__(self,c,dim):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 1)


        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(dim,dim,1)
        self.linear_2 = nn.Conv1d(self.k,c,1)
    

    def forward(self,x):

        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n

        x = self.linear_0(x) # b, k, n

        x = F.softmax(x, dim=-1) # b, k, n
        x = x / (1e-9 + x.sum(dim=1, keepdim=True)) #  b, k, n

        x = x.permute(0,2,1)  #b,n,k
        x = self.linear_1(x)
        x = x.permute(0,2,1) #b,k,n

        x = self.linear_2(x)

        x = x.permute(0,2,1)

        return x
