import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

class SpikeSA(nn.Module):
    """脉冲自注意力模块"""
    def __init__(self, embed_dim, num_heads, T=4):
        super().__init__()
        self.T = T  # 时间步长
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 脉冲化QKV投影
        self.q_proj = layer.SeqToANNContainer(nn.Linear(embed_dim, embed_dim))
        self.k_proj = layer.SeqToANNContainer(nn.Linear(embed_dim, embed_dim))
        self.v_proj = layer.SeqToANNContainer(nn.Linear(embed_dim, embed_dim))
        
        # 脉冲神经元
        self.spike = neuron.LIFNode(tau=2.0, detach_reset=True, surrogate_function='atan')
        
        # 输出投影
        self.out_proj = layer.SeqToANNContainer(nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        # x shape: [T, B, N, C]
        T, B, N, C = x.shape
        
        # 时序展开
        q = self.q_proj(x)  # [T,B,N,C]
        k = self.k_proj(x)  # [T,B,N,C]
        v = self.v_proj(x)
        
        # 多头处理
        q = q.reshape(T, B, N, self.num_heads, self.head_dim).permute(1,2,0,3,4) # [B,N,T,h,d]
        k = k.reshape(T, B, N, self.num_heads, self.head_dim).permute(1,2,3,4,0) # [B,N,h,d,T]
        v = v.reshape(T, B, N, self.num_heads, self.head_dim).permute(1,2,3,0,4) # [B,N,h,T,d]
        
        # 脉冲注意力得分
        attn = torch.einsum('bnthd,bnhdt->bntht', q, k) / torch.sqrt(torch.tensor(self.head_dim))
        attn = self.spike(attn)  # [B,N,T,h,T]
        
        # 脉冲加权聚合
        out = torch.einsum('bntht,bnhtd->bnthd', attn, v)  # [B,N,T,h,d]
        out = out.permute(2,0,1,3,4).reshape(T, B, N, C)  # [T,B,N,C]
        
        return self.out_proj(out)

class SpikeFFN(nn.Module):
    """脉冲前馈网络"""
    def __init__(self, embed_dim, expand_ratio=4):
        super().__init__()
        self.net = nn.Sequential(
            layer.SeqToANNContainer(nn.Linear(embed_dim, embed_dim*expand_ratio)),
            neuron.LIFNode(tau=2.0, surrogate_function='sigmoid'),
            layer.SeqToANNContainer(nn.Linear(embed_dim*expand_ratio, embed_dim)),
            neuron.LIFNode(tau=2.0, detach_reset=True)
        )
        
    def forward(self, x):
        return self.net(x)

class SpikeTransformerBlock(nn.Module):
    """脉冲Transformer块，带膜电位残差"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = SpikeSA(embed_dim, num_heads)
        self.ffn = SpikeFFN(embed_dim)
        self.mem_norm = neuron.LIFNode(tau=1e3, v_threshold=1e6)  # 膜电势存储器
        
    def forward(self, x):
        # 膜电势残差
        residual = x
        x = self.attn(x) + self.mem_norm(residual)
        x = self.ffn(x) + self.mem_norm(residual)
        return x

class SDSA(nn.Module):
    """堆叠脉冲自注意力模型"""
    def __init__(self, embed_dim, num_heads, num_layers, T=4):
        super().__init__()
        self.T = T
        self.layers = nn.ModuleList([
            SpikeTransformerBlock(embed_dim, num_heads) 
            for _ in range(num_layers)
        ])
        functional.set_step_mode(self, 'm')
        
    def forward(self, x):
        # 输入展开时序维度 [B,N,C] -> [T,B,N,C]
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1)
        
        for layer in self.layers:
            x = layer(x)
            
        # 时间维度平均
        return x.mean(0)  # [B,N,C]

class snn_vad(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 输入特征编码
        self.encoder = nn.Sequential(
            nn.Linear(args.f_feature_size, args.hid_dim),
            layer.MultiStepDropout(args.dropout)
        )
        
        # 脉冲Transformer
        self.sdsa = SDSA(
            embed_dim=args.hid_dim,
            num_heads=args.nhead,
            num_layers=args.n_transformer_layer,
            T=args.timesteps
        )
        
        # MIL分类器（保持原结构）
        self.MIL = MIL(args.hid_dim)

    def forward(self, f_f, seq_len=None):
        f_f = self.encoder(f_f)  # [B,N,C]
        sdsa_out = self.sdsa(f_f)  # [B,N,C]
        return self.MIL(sdsa_out, seq_len)

class MIL(nn.Module):
    """多示例学习分类器（与原实现一致）"""
    def __init__(self, input_dim, dropout_rate=0.6):
        super().__init__()
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
        instance_logits = torch.zeros(0).to(logits.device)
        for i in range(logits.size(0)):
            if seq_len is None:
                return logits
            k = int(seq_len[i]//16 + 1)
            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=k, largest=True)
            instance_logits = torch.cat((instance_logits, tmp.mean().unsqueeze(0)))
        return instance_logits

    def forward(self, avf_out, seq_len):
        avf_out = self.regressor(avf_out).squeeze(-1)
        return self.filter(avf_out, seq_len)