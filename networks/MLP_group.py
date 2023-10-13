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

        """
        这是全局的注意力block
        """

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


class localattention(nn.Module):
    def __init__(self,c,dim,ws = 1):
        super(localattention,self).__init__()
        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Conv2d(c,c,1)
        self.k = 64 
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_2 = nn.Linear(ws*ws,ws*ws)
        self.ws = ws #窗口的大小

    def forward(self,input):
        x = self.conv1(input)
        
        b, c, h, w = x.size()
        h_group,w_group = h//self.ws,w//self.ws

        total_groups = h_group*w_group
        x = x.view(b, c, h*w)   # b * c * n

        #这是第一部分
        x = self.linear_0(x) # b, k, n

        x = F.softmax(x, dim=-1) # b, k, n
        x = x / (1e-9 + x.sum(dim=1, keepdim=True)) #  b, k, n

        #第二部分局部注意力

        x = x.reshape(b,h_group,self.ws,w_group,self.ws,self.k)

        x = x.view(b,total_groups,-1,self.k)   #b,total_groups,(ws*ws),k
        x = x.permute(0,1,3,2)#b,total,k,ws*ws
        x = self.linear_2(x)
        x = x.permute(0,1,3,2) #b,total,ws*ws,k
        x = x.reshape(b,h_group,self.ws,w_group,self.ws,self.k)
        x = x.reshape(b,-1,self.k).permute(0,2,1) #b,n,k

        #最后一个部分
        x = self.linear_1(x)
        x = x.permute(0,2,1)

        return x


class GLAttention(nn.Module):
    def __init__(self,c,dim,ws = 8):
        super().__init__()
        self.ws = ws
        self.globalattn = external_attention2(c,dim)
        self.localattn = localattention(c,dim)
    
    def forward(self,x):
        if self.ws==1:
            x = self.globalattn(x)
        else:
            x = self.localattn(x)
        
        return x