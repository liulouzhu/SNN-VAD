import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from Transformer import *

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]
    

class ann_vad(nn.Module):
    def __init__(self, args):
        super(ann_vad, self).__init__()
        dropout = args.dropout
        nhead = args.nhead
        hid_dim = args.hid_dim
        ffn_dim = args.ffn_dim
        n_transformer_layer = args.n_transformer_layer
        
        # 只保留光流特征处理
        self.fc_f = nn.Linear(args.f_feature_size, hid_dim)
        self.msa = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), 
                                     PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        
        # 使用单模态Transformer处理
        self.mm_transformer = MultilayerTransformer(TransformerLayer(hid_dim, 
                                                  MultiHeadAttention(h=nhead, d_model=hid_dim), 
                                                  PositionwiseFeedForward(hid_dim, hid_dim), 
                                                  dropout), n_transformer_layer)
        
        # MIL分类器
        self.MIL = MIL(hid_dim)

    def forward(self, f_f, seq_len=None):  # 只接收光流特征
        # 处理光流特征
        f_f = self.fc_f(f_f)
        f_f = self.msa(f_f)
        
        # 使用单模态Transformer
        f_f = self.mm_transformer(f_f)

        print(f_f.shape)
        
        # 多示例学习分类
        MIL_logits = self.MIL(f_f, seq_len)
        
        # 不再需要计算对比损失
        return MIL_logits # 返回预测结果和零损失
    
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