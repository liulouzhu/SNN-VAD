import torch
import torch.nn as nn
import torch.nn.functional as F
from Spiking import LocalSpikingFeature, GlobalSpikingFeature, TemporalInteractionModule

class MIL(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.6):
        super(MIL, self).__init__()
        self.regressor = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(dropout_rate),
                                       nn.Linear(512, 32), nn.Dropout(dropout_rate),
                                       nn.Linear(32, 1), nn.Sigmoid())

    def filter(self, logits, seq_len):
        instance_logits = torch.zeros(0).cuda()
        for i in range(logits.shape[0]):
            if seq_len is None:
                return logits
            else:
                tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
                tmp = torch.mean(tmp).view(1)
            instance_logits = torch.cat((instance_logits, tmp))
        return instance_logits

    def forward(self, avf_out, seq_len):
        avf_out = self.regressor(avf_out)
        avf_out = avf_out.squeeze()
        mmil_logits = self.filter(avf_out, seq_len)
        return mmil_logits
    
class snn_vad(nn.Module):
    def __init__(self, args):
        super(snn_vad, self).__init__()
        feature_size = args.f_feature_size
        hid_dim = args.hid_dim
        
        # 光流特征处理
        self.lsf = LocalSpikingFeature(feature_size, feature_size//3)
        self.gsf = GlobalSpikingFeature(feature_size, feature_size//4)
        self.tim = TemporalInteractionModule(feature_size+feature_size//4, alpha=0.6)
        
        # MIL分类器
        self.MIL = MIL(hid_dim)

    def forward(self, f_f, seq_len=None):  # 只接收光流特征
        
        f_l = self.lsf(f_f)
        f_g = self.gsf(f_f)
        fused = torch.cat([f_l, f_g], dim=-1)
        # 使用时间交互模块
        tim_out = self.tim(fused)

        # 多示例学习分类
        MIL_logits = self.MIL(tim_out, seq_len)
        
        # 不再需要计算对比损失
        return MIL_logits