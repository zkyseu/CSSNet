# encoding: utf-8
import warnings
import numpy as np
import nibabel as nib
from functools import partial
import torch
import torch.nn as nn
from .Deformable_mlp import CycleMLP_B4_feat,CycleMLP_B5_feat,CycleMLP_B2_feat
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from .Coordconv import CoordConv
from mish_cuda import MishCuda as Mish
from .as_mlp import AS_MLP
from torchvision import transforms



from . import vit_seg_configs as configs




def get_model(name):
    assert name=='B4'or 'B5','name should be B4 or B5'
    model_cfg = {'B4':CycleMLP_B4_feat,'B5':CycleMLP_B5_feat,'B2':CycleMLP_B2_feat,'AS':AS_MLP}
    model = model_cfg[name]
    return model

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Mlp1(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x




class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super(Upsample, self).__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)


class External_attention(nn.Module):
   '''
   Arguments:
       c (int): The input and output channel number.
   '''
   def __init__(self, c):
       super(External_attention, self).__init__()
      
       self.conv1 = nn.Conv2d(c, c, 1)

       self.k = 64
       self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

       self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)        

   def forward(self, x):
       idn = x
       # The first 1x1 conv
       x = self.conv1(x)

       b, c, h, w = x.size()
       n = h*w
       x = x.view(b, c, h*w)   # b * c * n

       x = self.linear_0(x) # b, k, n
       x = F.softmax(x, dim=-1) # b, k, n
       x = x / (1e-9 + x.sum(dim=1, keepdim=True)) #  b, k, n
       x = self.linear_1(x) # b, c, n

       x = x.view(b, c, h, w)
       # The second 1x1 conv
       x = x + idn
       return x


class selfattn(nn.Module):
    def __init__(self,dim):
        super(selfattn,self).__init__()
        self.conv1 = nn.Conv2d(dim,dim,1,1)
        self.conv2 = nn.Conv2d(dim,dim,1,1)
        self.conv3 = nn.Conv2d(dim,dim,1,1)
        self.norm1 = torch.norm
        self.norm2 = torch.norm
    
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x2 = self.norm1(x2)
        x3 = self.norm2(x3)
        x1_2 = x1*x2.permute(0,1,3,2)
        out = x1_2*x3
        return out

class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            act = 'relu',
            use_batchnorm=True,):
        super(Conv2dReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        self.relu = nn.ReLU(inplace = True)

        self.bn = nn.BatchNorm2d(out_channels)

        self.mish = Mish()

        self.act = act

        
    
    def forword(self,x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act == 'relu':
            x = self.relu(x)
        else:
            x = self.mish(x)
        
        return x

class SRModel(nn.Module):
    """
    Proposed self-refinement model
    """
    def __init__(self,in_channel,em_channel):
        super(SRModel,self).__init__()
        self.conv1 = nn.Conv2d(in_channel,em_channel,3,1,1)
        self.bn1 = nn.BatchNorm2d(em_channel)
        self.conv2 = nn.Conv2d(em_channel,em_channel*2,1,1)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.em_channel = em_channel
    
    def forward(self,x):
        out = self.act1((self.bn1(self.conv1(x))))
        out1 = self.conv2(out)
        w,b = out1[:,:self.em_channel,:,:].clone(), out1[:,self.em_channel:,:,:].clone()
        return self.act2(w*out+b)

class MLPhead(nn.Module):
    """
    here we use a very simple mlp-based head to get the final output
    """
    def __init__(self,feature_strides,in_channel,num_class,embedding_dim=512):
        super(MLPhead,self).__init__()
        self.in_channel = in_channel
        assert min(feature_strides) == feature_strides[0]
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channel
       
        self.feature_strides = feature_strides

        self.embedding_dim = embedding_dim
        self.num_classes = num_class
        self.input_transform='multiple_select'
        self.in_index=[0, 1, 2, 3]
        self.dropout = nn.Dropout2d(0.1)

        # self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        # self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        # self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        # self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_c4 = CoordConv(in_channels=c4_in_channels, out_channels=embedding_dim,kernel_size=3,stride = 1,padding = 1)
        self.linear_c3 = CoordConv(in_channels=c3_in_channels, out_channels=embedding_dim,kernel_size=3,stride = 1,padding = 1)
        self.linear_c2 = CoordConv(in_channels=c2_in_channels, out_channels=embedding_dim,kernel_size=3,stride = 1,padding = 1)
        self.linear_c1 = CoordConv(in_channels=c1_in_channels, out_channels=embedding_dim,kernel_size=3,stride = 1,padding = 1)        

        self.linear_fuse = Conv2dReLU(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
          #  norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.attention = nn.ModuleList([selfattn(dim=512) for i in range(4)])

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        self.pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        #self.att = External_attention(c = 768)
    
    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs,origin_input):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4)
        #_c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)


        _c3 = self.linear_c3(c3)
        #_c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)


        _c2 = self.linear_c2(c2)
        #_c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)


        _c1 = self.linear_c1(c1)
        #_c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = resize(_c1, size=c1.size()[2:],mode='bilinear',align_corners=False)
        
        x = [_c4,_c3,_c2,_c1]
        _x = torch.cat(x,dim = 1)

        out = []
        for i in range(len(x)):
            out.append(self.attention[i](x[i]))
        
        
        _c = self.linear_fuse(torch.cat(out, dim=1)+_x)
#        c_att = self.att(_c)
#        _c+=c_att


        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = resize(x, size=origin_input.size()[2:],mode='bilinear',align_corners=False)
        x = self.linear_pred(x)

        return x,c2,c3,c4


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
            use_enhance = False
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            act = 'mish'
        )
        self.conv2 = CoordConv(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.act = Mish()
        self.bn = nn.BatchNorm2d(out_channels)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(self.bn(x))
        return x

class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes)+1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear',align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class ConvBlock(nn.Module):
    def __init__(self,in_channel,out_channels,use_coord = False):
        super(ConvBlock,self).__init__()
        self.conv1_1 = Conv2dReLU(in_channel,out_channels,kernel_size = 1,stride = 1,use_batchnorm = True)
        self.conv3_3 = Conv2dReLU(out_channels,out_channels,kernel_size = 3,stride = 1, padding = 1, use_batchnorm = True,act = 'mish')
        self.use_coord = use_coord
        self.coord3_3 = nn.Sequential(CoordConv(out_channels,out_channels,kernel_size=3,stride=1,padding = 1),nn.BatchNorm2d(out_channels),Mish())
    
    def forward(self,x):
        x = self.conv1_1(x)
        if self.use_coord:
            x = self.coord3_3(x)
        else:
            x = self.conv3_3(x)
        
        return x

class CSPBlock(nn.Module):
    def __init__(self,in_channel):
        super(CSPBlock,self).__init__()
        self.conv1 = nn.Sequential(CoordConv(in_channel,in_channel//2,kernel_size=1,stride=1),nn.BatchNorm2d(in_channel//2),Mish())
        self.conv2 = nn.Sequential(CoordConv(in_channel,in_channel//2,kernel_size=1,stride=1),nn.BatchNorm2d(in_channel//2),Mish())
        self.conv_block = nn.Sequential(ConvBlock(in_channel//2,in_channel//2),ConvBlock(in_channel//2,in_channel//2,use_coord = True),ConvBlock(in_channel//2,in_channel//2))
        self.conv3 = nn.Sequential(CoordConv(in_channel,in_channel,kernel_size=1,stride=1),nn.BatchNorm2d(in_channel),Mish())
    
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.conv_block(x2)
        out = self.conv3(torch.cat([x1,x2],dim = 1))
        return out 


class DownDecoder(nn.Module):
    def __init__(self,in_channel,out_channels,skip_channels,use_batchnorm=True):
        super(DownDecoder,self).__init__()
        self.PPN = PSPModule(features = in_channel//4,out_features = in_channel)
        self.act1 = Mish()
        self.act2 = Mish()
        self.SR = SRModel(out_channels,out_channels)
        self.conv1 = Conv2dReLU(
            in_channel + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            act = 'mish',
        )
        self.conv2 = CoordConv(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = CoordConv(in_channel,in_channel,kernel_size=3,stride = 1,padding = 1)
        self.bn3 = nn.BatchNorm2d(in_channel)
        self.down = Conv2dReLU(in_channel,in_channel,kernel_size=3,stride=2,padding = 1,use_batchnorm = use_batchnorm,act = 'mish')
        self.outconv = Conv2dReLU(out_channels,192,kernel_size=3,stride=1,padding = 1,use_batchnorm = use_batchnorm)
        self.outcoord = nn.Sequential(CoordConv(out_channels,192,kernel_size=3,stride=1,padding = 1),nn.BatchNorm2d(192),Mish())
        self.CSPBlock = CSPBlock(out_channels)

    def forward(self,inputs,skip = None):
        x = self.down(inputs)
        if skip is not None:
            x = torch.cat([x,skip],dim = 1)
        x = self.conv1(x)
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.CSPBlock(x)
        x = self.SR(x)
        out = x
        out = self.outcoord(out)
        return x,out


class PANetMedical(nn.Module):
    def __init__(self,in_channel,num_class = 4,embed_dim = 512,use_batchnorm = True):
        super(PANetMedical,self).__init__()
        self.attention = nn.ModuleList([External_attention(c=embed_dim//4) for i in range(4)])
        self.up_out_channel = in_channel[1:]
        self.down_out_channel = in_channel[::-1][1:]
        self.up_model = nn.ModuleList([DecoderBlock(in_ch, out_ch, sk_ch)  for in_ch, out_ch, sk_ch in zip(in_channel[:-1], self.up_out_channel, self.up_out_channel)])
        self.down_model = nn.ModuleList([DownDecoder(in_ch, out_ch, sk_ch)  for in_ch, out_ch, sk_ch in zip(in_channel[::-1][:-1], self.down_out_channel, self.down_out_channel)])
        self.conv = Conv2dReLU(embed_dim,embed_dim,kernel_size=3,stride=1,padding = 1,use_batchnorm = use_batchnorm,act = 'mish')
        self.conv1 = Conv2dReLU(in_channel[0],embed_dim//4,kernel_size=3,stride=1,padding = 1,use_batchnorm = use_batchnorm)
        self.conv2 = Conv2dReLU(embed_dim,embed_dim,kernel_size=3,stride=1,padding = 1,use_batchnorm = use_batchnorm)
        self.dropout = nn.Dropout2d(0.1)
        self.pred = nn.Conv2d(embed_dim, num_class, kernel_size=1)
        self.PSPmodel = PSPModule(embed_dim,embed_dim)

    def forward(self,inputs,origin_input):
        c1, c2, c3, c4 = inputs # 1/4,1/8,1/16,1/32
        #c4 = self.PSPmodel(c4)
        up_out = []
        down_out = []
        c4 = self.conv(c4)
        input = [c4,c3,c2,c1]
        #up path
        for i,block in enumerate(self.up_model):
            if i == 0:
                out = block(input[i],input[i+1])
            else:
                out = block(out,input[i+1])
            up_out.append(out)
        up_out = up_out[::-1]
        #down path
        for i,block in enumerate(self.down_model):
            if i == 0:
                out,final_out = block(up_out[i],up_out[i+1])
                final_out = resize(final_out, size=c1.size()[2:],mode='bilinear',align_corners=False)
                down_out.append(final_out)
            elif i==2:
                out,final_out = block(up_out[i],c4)
                final_out = resize(final_out, size=c1.size()[2:],mode='bilinear',align_corners=False)
                down_out.append(final_out)
            else:
                out,final_out = block(out,up_out[i+1])
                final_out = resize(final_out, size=c1.size()[2:],mode='bilinear',align_corners=False)
                down_out.append(final_out)
        out1 = self.conv1(out)
        out1 = resize(out1, size=c1.size()[2:],mode='bilinear',align_corners=False)
        out = down_out + [out1]
        _out = torch.cat(out,dim = 1)
        out = [self.attention[i](out[i]) for i in range(4)]
        out = torch.cat(out,dim=1)+_out
        out = self.dropout(out)
        out = resize(out, size=origin_input.size()[2:],mode='bilinear',align_corners=False)
        out = self.pred(out)
        return out,c2,c3,c4

        
        


class MLPmedical(nn.Module):
    def __init__(self,config,num_class = 6):
        super(MLPmedical,self).__init__()
        self.model = get_model('AS')()
        self.config = config
        self.num_class = num_class
      #  self.config['decoder_channels'] = (768,384,192,96)
        self.in_channel = (768,384,192,96)
        self.feature_stride = [4, 8, 16, 32]
        self.seghead = PANetMedical(in_channel=self.in_channel,num_class = self.num_class,embed_dim = 768)
        self.segmentation_head = SegmentationHead(
            in_channels=self.in_channel[-1],
            out_channels=num_class,
            kernel_size=3,
        )

        self.segmentation_head2 = SegmentationHead(
            in_channels=self.in_channel[-2],
            out_channels=self.in_channel[-1],
            kernel_size=3,
            )
        self.segmentation_head1 = SegmentationHead(
            in_channels=self.in_channel[-3],
            out_channels=self.in_channel[-2],
            kernel_size=3,
            )
        self.segmentation_head0 = SegmentationHead(
            in_channels=self.in_channel[-4],
            out_channels=self.in_channel[-3],
            kernel_size=3,
            )
    
    def tensor_to_PIL(self,tensor):
    #unloader = transforms.ToPILImage()
        unloader = transforms.ToPILImage()
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        image = unloader(image)
        return image

    def viz(self,x,ind):
        _,C,H,W = x.shape
        img = np.zeros([H,W,C])
        for i in range(C):
            temp = self.tensor_to_PIL(x[:,i,:,:])
            img[:,:,i] = np.array(temp)
        affine = np.diag([1,2,3,1])
        img1=nib.Nifti1Image(img,affine)
        nib.save(img1,'/home/kkk/medical/TransUNet/save_nii/my_file_{}.nii.gz'.format(ind))
           


    def forward(self,inputs,test = None,ind = None):
        if inputs.size()[1] == 1:
            inputs = inputs.repeat(1,3,1,1)

        out = self.model(inputs)
        out = tuple(out)
        logits,c2,c3,c4 = self.seghead(out,inputs)
        logits2 = self.segmentation_head2(c2)
        logits2 = self.segmentation_head(logits2)
        logits1 = self.segmentation_head1(c3)
        logits1 = self.segmentation_head2(logits1)
        logits1 = self.segmentation_head(logits1)
        logits0 = self.segmentation_head0(c4)
        logits0 = self.segmentation_head1(logits0)
        logits0 = self.segmentation_head2(logits0)
        logits0 = self.segmentation_head(logits0)
        if test:
            self.viz(logits,ind = ind)

        return logits,logits2,logits1,logits0


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}





if __name__=='__main__':
    
    img = torch.ones([1, 3, 224, 224])
    
    model = MLPmedical(CONFIGS['R50-ViT-B_16'])
   # snapshot = '/home/kkk/medical/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo250_bs12_lr0.005_224/epoch_149.pth'
   # model.load_state_dict(torch.load(snapshot))
    logits,logits2 = model(img)
    out_list = [logits,logits2]
    for log in out_list:
        print(log.shape)

