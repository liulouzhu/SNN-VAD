import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional


class LIFNeuron(neuron.LIFNode):
    def __init__(self, tau=10.0, v_threshold=1.0, v_reset=0.0, detach_reset=True):
        # 添加 detach_reset 参数，断开重置操作的计算图
        super().__init__(tau=tau, v_threshold=v_threshold, v_reset=v_reset, detach_reset=detach_reset)
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset

class LocalSpikingFeature(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5], dilations=[1, 2]):
        super().__init__()
        self.conv_blocks = nn.ModuleList()
        for k, d in zip(kernel_sizes, dilations):
            padding = (k - 1) * d // 2
            self.conv_blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=k, 
                             dilation=d, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    LIFNeuron(tau=10.0)
                )
            )
    
    def forward(self, x):
        # 输入形状: [T, B, C]
        x = x.permute(1, 2, 0)  # [B, C, T]
        features = []
        for conv in self.conv_blocks:
            out = conv(x)
            features.append(out)
        concat_feat = torch.cat(features, dim=1)  # [B, 3*C, T]
        return concat_feat.permute(2, 0, 1)  # [T, B, 3*C]

class GlobalSpikingFeature(nn.Module):
    def __init__(self, in_channels, out_channels, sigma=1.0):
        super().__init__()
        self.sigma = sigma
        self.channel_reduce = nn.Conv1d(in_channels, out_channels, 1)
        self.gcn_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.lif = LIFNeuron(tau=10.0)
        
    def build_adjacency(self, x):
        # 特征相似度分支
        norm_x = F.normalize(x, p=2, dim=1)
        sim_matrix = torch.bmm(norm_x.transpose(1,2), norm_x)  # [B, T, T]
        
        # 位置距离分支
        T = x.size(2)
        pos = torch.arange(T, device=x.device, dtype=torch.float)
        pos_diff = pos.view(1, T, 1) - pos.view(1, 1, T)
        dis_matrix = torch.exp(-torch.abs(pos_diff) / self.sigma)  # [1, T, T]
        
        sim_softmax = F.softmax(sim_matrix, dim=-1)
        dis_softmax = F.softmax(dis_matrix, dim=-1)
        combined = sim_softmax + dis_softmax
        
        return combined

    def forward(self, x):
        # 输入形状: [T, B, C]
        x = x.permute(1, 2, 0)  # [B, C, T]
        reduced = self.channel_reduce(x)  # [B, C', T]
        
        adj = self.build_adjacency(reduced)  # [B, T, T]

        gcn_out = self.gcn_conv(reduced)     # [B, C', T]
        gcn_out = torch.bmm(gcn_out, adj)    # [B, C', T]
        
        return self.lif(gcn_out.permute(2, 0, 1))  # [T, B, C']

class TemporalInteractionModule(nn.Module):
    def __init__(self, channels, alpha=0.6):
        super().__init__()
        self.alpha = alpha
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.BatchNorm1d(channels),
            LIFNeuron(tau=10.0)
        )
        
    def forward(self, x):
        # 输入形状: [T, B, C]
        T, B, C = x.shape
        x = x.permute(1, 2, 0)  # [B, C, T]
        
        outputs = []
        prev_state = torch.zeros(B, C, 1, device=x.device)
        for t in range(T):
            current = x[:, :, t].unsqueeze(-1)  # [B, C, 1]
            
            # 时间交互
            conv_state = self.temporal_conv(prev_state)
            new_state = (1-self.alpha)*current + self.alpha*conv_state
            outputs.append(new_state.squeeze(-1))
            
            prev_state = new_state.detach().clone()
            
        return torch.stack(outputs, dim=0)  # [T, B, C]