import torch
import torch.nn as nn

import torch.nn.functional as F
from torch import einsum
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

from einops.layers.torch import Rearrange
from einops import rearrange

from .attmlp import External_attention
from .attmlp import Gmlp

from .Vit_new import Convmpl
from .Vit_new import SegmentationHead
from .Vit_new import CONFIGS
from .Vit_new import DecoderCup

from .functions.deform_conv import DeformConv2D
from .local_attention import localattention

class SepConv2d(torch.nn.Module):
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


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class loacl_feature(nn.Module):
    def __init__(self,dim,hidden_dim):
        super(loacl_feature,self).__init__()
        self.maxpool = nn.MaxPool2d(3,2,1)
        self.conv1 = nn.Conv2d(dim,hidden_dim,1,1)
        self.conv2 = nn.Conv2d(hidden_dim,hidden_dim,3,1,1)
        self.part3 = nn.Sequential(nn.AdaptiveAvgPool2d(1,1),nn.Conv2d(hidden_dim,hidden_dim,1,1))

    def forward(self,input):
        x = self.conv1  
        out = self.conv2(x)+self.part3(x)+self.maxpool(x)

#条件位置编码
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim), )
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class enhancefeature(nn.Module):
    """
    高层语义信息增强模块
    """
    def __init__(self,in_channel):
        super(enhancefeature,self).__init__()
        self.conv3_3 = nn.Conv2d(in_channel,in_channel//2,3,1,1)
        self.conv5_5 = nn.Conv2d(in_channel,in_channel//2,5,1,2)
        self.conv1_1 = nn.Conv2d(in_channel//8,in_channel,1,1)
        self.glob = nn.AdaptiveAvgPool2d((1,1))
        self.conv1_1_2 = nn.Conv2d(in_channel,in_channel//8,1,1)
        self.up1 = nn.PixelShuffle(2)
        self.up2 = nn.PixelShuffle(2)
    
    def forward(self,x):
        x1 = self.up1(self.conv3_3(x))
        x2 = self.up2(self.conv5_5(x))
        x3 = self.conv1_1_2(self.glob(x))
        out = x1+x2+x3
        out = self.conv1_1(out)
        return out


class LeFF(nn.Module):
    
    def __init__(self, h,w,dim = 192, scale = 4, depth_kernel = 3,use_deform = False):
        super().__init__()
        
        scale_dim = dim*scale
        self.use_deform = use_deform
        self.up_proj = nn.Sequential(nn.Linear(dim, scale_dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(scale_dim),
                                    nn.GELU(),
                                    Rearrange('b c (h w) -> b c h w', h=h, w=w)
                                    )
        
        self.depth_conv =  nn.Sequential(SepConv2d(scale_dim, scale_dim, kernel_size=depth_kernel, padding=1),
                          nn.BatchNorm2d(scale_dim),
                          nn.GELU(),
                          Rearrange('b c h w -> b (h w) c', h=h, w=w)
                          )
        
        self.down_proj = nn.Sequential(nn.Linear(scale_dim, dim),
                                    Rearrange('b n c -> b c n'),
                                    nn.BatchNorm1d(dim),
                                    nn.GELU(),
                                    Rearrange('b c n -> b n c')
                                    )
        self.offset = nn.Conv2d(scale_dim,2 * 3 * 3,kernel_size=(3, 3),stride=(1, 1),padding=(1, 1),bias=False)

        
        self.norm = nn.Sequential(nn.BatchNorm2d(scale_dim),
                                  nn.GELU(),
                                  Rearrange('b c h w -> b (h w) c', h=h, w=w)
                                  )

        self.deformConv = DeformConv2D(scale_dim,scale_dim, kernel_size = 3,padding=1)
                                        

    def forward(self, x):
        x = self.up_proj(x)
        if self.use_deform:
            offset = self.offset(x)
            x = self.deformConv(x,offset)
            x = self.norm(x)
        else:
            x = self.depth_conv(x)
        x = self.down_proj(x)
        return x

class Mlp(nn.Module):
    """
    多层感知器
    """
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



class Attention(nn.Module):
    """
    this is self-attention in transformer
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,kernel_size=3,
                 q_stride = 1,k_stride = 2,v_stride = 2,pad = 1,use_convproj = True,
                 use_external = None,dim_n = None,use_fc = None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        #here is our modify
        self.use_convproj = use_convproj
        self.to_q = SepConv2d(dim, dim, kernel_size, q_stride, pad)
        self.to_k = SepConv2d(dim, dim, kernel_size, k_stride, pad)
        self.to_v = SepConv2d(dim, dim, kernel_size, v_stride, pad)

        #here is our second modify
        self.use_external = use_external
        self.externel_attention = External_attention(dim,dim_n,use_fc = use_fc)

        #here is our third modify
        self.local_attention = localattention(dim,dim_n,ws = 8)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        if self.use_external:
            x = rearrange(x, 'b (l w) n -> b n l w', l=H, w=W)
            x = self.externel_attention(x)
            #x = self.local_attention(x)
        else:    
            if self.use_convproj:
                x = rearrange(x, 'b (l w) n -> b n l w', l=H, w=W)
                q = self.to_q(x)
                q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=self.num_heads)

                v = self.to_v(x)
                v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=self.num_heads)

                k = self.to_k(x)
                k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=self.num_heads)

                attn = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = einsum('b h i j, b h j d -> b h i d', attn, v)

                x = rearrange(x, 'b h n d -> b n (h d)')
                x = self.proj(x)
                x = self.proj_drop(x)



            else:
                q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                if self.sr_ratio > 1:
                    x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                    x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                    x_ = self.norm(x_)
                    kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                else:
                    kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k, v = kv[0], kv[1]

                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x)
                x = self.proj_drop(x)

        return x

class Block(nn.Module):

    """
    this is one transformer block 
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,use_local = False,
                 width = None,use_external = None,dim_n = None,use_fc = False,use_deform = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.h = self.w = width
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
            use_external=use_external,dim_n=dim_n,use_fc=use_fc)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.hidden_dim = mlp_hidden_dim
        self.ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.use_local = use_local
        self.conv_mpl = Residual(PreNorm(dim, LeFF(self.h,self.w,dim, mlp_ratio,use_deform=use_deform)))
        self.use_external = use_external
        self.gmlp = Gmlp(dim_n)
        # self.local = loacl_feature(dim,mlp_hidden_dim)
       # self.convmpl = Convmpl(dim,mlp_hidden_dim)   # here is our modify

    def forward(self, x, H, W):

        if not self.use_external:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        else:
            x= x+self.drop_path(self.attn(self.norm1(x),H,W))
#            x = x + self.gmlp(self.norm1(x).permute(0,2,1)).permute(0,2,1)

        
        # if self.use_local:
        #     x = 
        #x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.conv_mpl(x)
        #x=  x + self.drop_path(self.convmpl(self.norm2(x)))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,use_depth = True):
        super().__init__()
        img_size = to_2tuple(img_size) #变成元组
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size) #这边就是降维操作
        if use_depth:
            self.depth_conv = nn.Conv2d(in_chans,embed_dim,kernel_size = patch_size,stride=patch_size,groups=in_chans) #使用深度可分离卷积降维
        # if use_depth:
        #     self.depth_conv = resnet_block(in_chans,embed_dim,patch_size) #使用深度可分离卷积降维
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        self.in_channel = in_chans
        self.use_depth = use_depth

    def forward(self, x):
        B, C, H, W = x.shape
        
        if self.use_depth:
            x = self.depth_conv(x).flatten(2).transpose(1, 2)
        else:
            x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class MultilayerVisonTransformer(nn.Module):
    """
    this is our proposed mlvit
    """
    def __init__(self,img_size=128, patch_size=2, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],dim_n = [64*64,32*32,16*16,8*8]):
        super(MultilayerVisonTransformer,self).__init__()
        self.num_class = num_classes #here is used for image classification
        self.depths = depths # here is similar to resnet which is repeat time for transformer block

        #here is four scales,embed_dim is out channel for each convolution
        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                       embed_dim=embed_dims[0],use_depth=False)
        self.patch_embed2 = PatchEmbed(img_size=img_size // 2, patch_size=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1],use_depth=False)
        self.patch_embed3 = PatchEmbed(img_size=img_size // 4, patch_size=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2],use_depth=False)
        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=2, in_chans=embed_dims[2],
                                       embed_dim=embed_dims[3],use_depth=False)
        
        #here is position embedding
        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, embed_dims[0]))
        self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, embed_dims[1]))
        self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, embed_dims[2]))
        self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, embed_dims[3]))
        self.pos_drop4 = nn.Dropout(p=drop_rate)


        #here is our modify position embedding
        self.peg1 = PosCNN(embed_dims[0],embed_dims[0])
        self.peg2 = PosCNN(embed_dims[1],embed_dims[1])
        self.peg3 = PosCNN(embed_dims[2],embed_dims[2])
        self.peg4 = PosCNN(embed_dims[3],embed_dims[3])


        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0],width=64,use_external=True,dim_n=dim_n[0])
            for i in range(depths[0])])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1],width=32,use_external=True,dim_n = dim_n[1])
            for i in range(depths[1])])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2],width=16,use_external=True,dim_n=dim_n[2])
            for i in range(depths[2])])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3],width=8,use_external=True,dim_n=dim_n[3])
            for i in range(depths[3])])
        self.norm = norm_layer(embed_dims[3])

        # init weights
        trunc_normal_(self.pos_embed1, std=.02)
        trunc_normal_(self.pos_embed2, std=.02)
        trunc_normal_(self.pos_embed3, std=.02)
        trunc_normal_(self.pos_embed4, std=.02)
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x):
        """
        here is structure of mlvit
        """
        B = x.shape[0]

        # stage 1
        x, (H, W) = self.patch_embed1(x)
        # x = x + self.pos_embed1
        x = self.pos_drop1(x)
        for i,blk in enumerate(self.block1):
            x = blk(x, H, W)
            if i==0:
                x = self.peg1(x,H,W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #[b,64,56,56]
        s1 = x

        # stage 2
        x, (H, W) = self.patch_embed2(x) #[b,786,128]
        # x = x + self.pos_embed2
        # x = x + self.pos_embed2  
        x = self.pos_drop2(x)
        for i,blk in enumerate(self.block2):
            x = blk(x, H, W)
            if i==0:
                x = self.peg2(x,H,W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #[b,128,28,28]
        s2 = x

        # stage 3
        x, (H, W) = self.patch_embed3(x)
        # x = x + self.pos_embed3
        x = self.pos_drop3(x)
        for i,blk in enumerate(self.block3):
            x = blk(x, H, W)
            if i==0:
                x = self.peg3(x,H,W)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #[b,256,14,14]
        s3 = x
        # stage 4
        x, (H, W) = self.patch_embed4(x)
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x = x + self.pos_embed4
        x = self.pos_drop4(x)
        for i,blk in enumerate(self.block4):
            x = blk(x, H, W)          #[b,49,512]
            if i==0:
                x = self.peg4(x,H,W)

        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #[b,512,7,7]
        s4 = x

        return [s1,s2,s3,s4]
    
    def forward(self, x):
        out = self.forward_features(x)

        return out


class MLViTSeg(nn.Module):
    def __init__(self,config):
        super(MLViTSeg,self).__init__()
        self.MLViT = MultilayerVisonTransformer()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)#上采样使用.

        self.decoder = DecoderCup(config)
#        self.transconv = nn.Sequential(nn.ConvTranspose2d(
#                in_channels=64,
#                out_channels=32,
#                kernel_size=4,
#                padding=1,
#                stride=2),
#            nn.ConvTranspose2d(
#                in_channels=32,
#                out_channels=16,
#                kernel_size=4,
#                padding=1,
#                stride=2),
#            nn.BatchNorm2d(num_features=16, momentum=0.9),
#            nn.ReLU(),)
        
        #语义分割头部分
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.segmentation_head2 = SegmentationHead(
            in_channels=config['decoder_channels'][-2],
            out_channels=config['decoder_channels'][-1],
            kernel_size=3,
        )
        self.segmentation_head1 = SegmentationHead(
            in_channels=config['decoder_channels'][-3],
            out_channels=config['decoder_channels'][-2],
            kernel_size=3,
        )
        self.segmentation_head0 = SegmentationHead(
            in_channels=config['decoder_channels'][-4],
            out_channels=config['decoder_channels'][-3],
            kernel_size=3,
        )


        self.config = config
    
    def forward(self,img):
        if img.size()[1] == 1:
            img = img.repeat(1,3,1,1)
        out_vit = self.MLViT(img)
        features = out_vit[:-1]
        features = features[::-1]

        x,feats = self.decoder(out_vit[-1], features) 
        #print('=========shape================',len(feats),feats[0].shape,x.shape)
        logits = self.segmentation_head(x)
        logits2 = self.segmentation_head2(feats[2])
        logits2 = self.segmentation_head(logits2)
        logits1 = self.segmentation_head1(feats[1])
        logits1 = self.segmentation_head2(logits1)
        logits1 = self.segmentation_head(logits1)
        logits0 = self.segmentation_head0(feats[0])
        logits0 = self.segmentation_head1(logits0)
        logits0 = self.segmentation_head2(logits0)
        logits0 = self.segmentation_head(logits0)
        return logits,logits2,logits1,logits0

        return logits,logits2,logits1,logits0


if __name__=='__main__':
    
    img = torch.ones([1, 3, 224, 224])
    
    model = MLViTSeg(CONFIGS['R50-ViT-B_16'])
   # snapshot = '/home/kkk/medical/model/TU_Synapse224/TU_pretrain_R50-ViT-B_16_skip3_epo250_bs12_lr0.005_224/epoch_149.pth'
   # model.load_state_dict(torch.load(snapshot))
    logits,logits2,logits1,logits0 = model(img)
    out_list = [logits,logits2,logits1,logits0]
    # for i in range(4):
        
    # model1 = MultilayerVisonTransformer()
    # out = model1(img)
    # for i in range(4):
    #     print('stage {} shape:'.format(i),out[i].shape)
    

