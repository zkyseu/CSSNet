import torch
import torch.nn as nn
import torch.nn.functional as F


class localattention(nn.Module):
    def __init__(self,c,dim,ws = 1):
        super(localattention,self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.k = 64 
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_0_0 = nn.Conv1d(c,self.k,1,bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False) 
        self.linear_2 = nn.Linear(ws*ws,ws*ws)
        self.ws = ws #窗口的大小

    def forward(self,x):
        x = self.conv1(x)
        ind = x

        b, c, h, w = x.size()
        h_group,w_group = h//self.ws,w//self.ws

        total_groups = h_group*w_group
        # x = x.view(b, c, h*w)   # b * c * n

        x = x.view(b,c,h*w)
        x = self.linear_0_0(x)
        x = x.reshape(b,h_group,self.ws,w_group,self.ws,self.k)

        x = x.view(b,total_groups,-1,self.k)   #b,total_groups,(ws*ws),k
        x = x.permute(0,1,3,2)#b,total,k,ws*ws

        #通道注意力机制(全局)
        ind = ind.view(b,c,h*w)
        ind = self.linear_0(ind)
        ind = F.softmax(ind, dim=-1) # b, k, n
        ind = ind / (1e-9 + ind.sum(dim=1, keepdim=True)) #  b, k, n

        #局部全连接(局部)
        x = self.linear_2(x)
        x = F.softmax(x, dim=-1)
        x = x / (1e-9 + x.sum(dim=1, keepdim=True)) # b*total_groups,k,(ws*ws)
        x = x.permute(0,1,3,2) #b,total,ws*ws,k
        x = x.reshape(b,h_group,self.ws,w_group,self.ws,self.k)
        x = x.reshape(b,-1,self.k).permute(0,2,1) #b,n,k

        out = x+ind
        out = self.linear_1(out)
        out = out.permute(0,2,1)

        return out
        









