import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

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
    

class snn_test(nn.Module):
    def __init__(self, args):
        super(snn_test, self).__init__()
        feature_size = args.f_feature_size
        hid_dim = args.hid_dim
        dropout = args.dropout
    
        self.fc_f = nn.Linear(feature_size, hid_dim)

        self.snn_seq = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hid_dim, hid_dim),
                snn.Leaky(beta=0.9, threshold=1.0),
                nn.Dropout(dropout)
            )for _ in range(2)
        ])

        # MIL分类器
        self.MIL = MIL(hid_dim)

    def forward(self, f_f, seq_len=None):
        f_f = self.fc_f(f_f)

        f_f = f_f.permute(1, 0, 2)  # [B, T, C] -> [T, B, C]

        mems = [block[1].init_leaky() for block in self.snn_seq]
        
        spk_out = []

        for t_step in f_f:
            x = t_step
            for i, block in enumerate(self.snn_seq):
                x = block[0](x)
                spk, mems[i] = block[1](x, mems[i])
                x = block[2](spk)
            spk_out.append(x.unsqueeze(0))

        f_f = torch.cat(spk_out).permute(1, 0, 2)  # [T, B, C] -> [B, T, C]

        MIL_logits = self.MIL(f_f, seq_len)

        return MIL_logits