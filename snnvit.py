import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from spikingjelly.clock_driven import layer
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from einops.layers.torch import Rearrange

class MLP(nn.Module):
    # 保持文档2中的MLP类不变
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.mlp1_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.mlp1_conv = nn.Conv2d(in_features, hidden_features, kernel_size=1, stride=1)
        self.mlp1_bn = nn.BatchNorm2d(hidden_features)

        self.mlp2_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.mlp2_conv = nn.Conv2d(hidden_features, out_features, kernel_size=1, stride=1)
        self.mlp2_bn = nn.BatchNorm2d(out_features)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.mlp1_lif(x)
        x = self.mlp1_conv(x.flatten(0, 1))
        x = self.mlp1_bn(x).reshape(T, B, self.c_hidden, H, W)

        x = self.mlp2_lif(x)
        x = self.mlp2_conv(x.flatten(0, 1))
        x = self.mlp2_bn(x).reshape(T, B, C, H, W)
        return x

class SpikingSelfAttention(nn.Module):
    # 保持文档2中的SpikingSelfAttention类不变
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads

        self.proj_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.q_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')

        self.attn_lif = MultiStepLIFNode(tau=1.5, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.proj_conv = nn.Conv1d(dim, dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_lif(x).flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        # 保持前向传播逻辑不变
        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4)

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N)
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0,1,3,2,4)

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125
        x = x.transpose(3,4).reshape(T, B, C, N)
        x = self.attn_lif(x).flatten(0,1)
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        return x

class SpikingTransformer(nn.Module):
    # 保持文档2中的SpikingTransformer类不变
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.attn = SpikingSelfAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim*mlp_ratio))

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class SpikingTokenizer(nn.Module):
    # 保持文档2中的SpikingTokenizer类不变
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = img_size_h//patch_size[0], img_size_w//patch_size[1]
        
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj1_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj1_conv = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims//4)

        self.proj2_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims//2)

        self.proj3_lif = MultiStepLIFNode(tau=1.5, detach_reset=True, backend='cupy')
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.proj3_conv = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0,1))
        x = self.proj_bn(x).reshape(T,B,-1,H,W)
        
        x = self.proj1_lif(x).flatten(0,1)
        x = self.maxpool1(x)
        x = self.proj1_conv(x)
        x = self.proj1_bn(x).reshape(T,B,-1,H//2,W//2)

        x = self.proj2_lif(x).flatten(0,1)
        x = self.maxpool2(x)
        x = self.proj2_conv(x)
        x = self.proj2_bn(x).reshape(T,B,-1,H//4,W//4)

        x = self.proj3_lif(x).flatten(0,1)
        x = self.maxpool3(x)
        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T,B,-1,H//8,W//8)
        return x, (H//8, W//8)

class vit_snn(nn.Module):
    # 修改后的vit_snn类，适配时序输出
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, embed_dims=256,
                 num_heads=8, mlp_ratios=4, depths=6):
        super().__init__()
        self.patch_embed = SpikingTokenizer(img_size_h=img_size_h, img_size_w=img_size_w,
                          patch_size=patch_size, in_channels=in_channels, embed_dims=embed_dims)
        self.blocks = nn.ModuleList([SpikingTransformer(embed_dims, num_heads, mlp_ratios) 
                                   for _ in range(depths)])

    def forward_features(self, x):
        x, (H, W) = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return x.mean(dim=-1).mean(dim=-1)  # 空间维度平均池化

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T,B,C,H,W]
        x = self.forward_features(x)  # 输出[T,B,D]
        return x.permute(1,0,2)       # [B,T,D]

class snnvit_vad(nn.Module):
    def __init__(self, args):
        super(snnvit_vad, self).__init__()
        # 特征维度转换
        self.fc_f = nn.Linear(args.f_feature_size, 2*args.img_size_h*args.img_size_w)
        
        # Spikingformer主体
        self.spikingformer = vit_snn(
            img_size_h=args.img_size_h,
            img_size_w=args.img_size_w,
            in_channels=2,
            embed_dims=args.hid_dim,
            num_heads=args.nhead,
            mlp_ratios=args.mlp_ratio,
            depths=args.n_transformer_layer
        )
        
        # 保留MIL分类器
        self.MIL = MIL(args.hid_dim)

    def forward(self, f_f, seq_len=None):
        # 输入转换 [B,T,D] -> [B,T,C,H,W]
        B, T, _ = f_f.shape
        f_f = self.fc_f(f_f).view(B, T, 2, self.args.img_size_h, self.args.img_size_w)
        
        # 脉冲神经网络处理 
        features = self.spikingformer(f_f)  # [B,T,D]
        
        # MIL分类
        return self.MIL(features, seq_len)

class MIL(nn.Module):
    # 完全保留文档1中的MIL实现
    def __init__(self, input_dim, dropout_rate=0.6):
        super(MIL, self).__init__()
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 32),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def filter(self, logits, seq_len):
        instance_logits = torch.zeros(0).cuda()
        for i in range(logits.shape[0]):
            if seq_len is None:
                return logits
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i]//16 +1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        return instance_logits

    def forward(self, avf_out, seq_len):
        avf_out = self.regressor(avf_out).squeeze()
        return self.filter(avf_out, seq_len)