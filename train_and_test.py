from sklearn.metrics import auc, precision_recall_curve
import torch
import numpy as np
import math
from tqdm import tqdm

def train(args, dataloader, model_MT, optimizer_MT, criterion, logger):
    with torch.set_grad_enabled(True):
        model_MT.train()
        for i, (f_f, label) in enumerate(dataloader):
            seq_len = torch.sum(torch.max(torch.abs(f_v), dim=2)[0] > 0, 1)
            f_f = f_f[:, :torch.max(seq_len), :]
            f_f, label = f_f.float().cuda(), label.float().cuda()
            MIL_logits = model_MT(f_f, seq_len)
            loss_MIL = criterion(MIL_logits, label)
            total_loss = loss_MIL
            logger.info(f"Current batch: {i}, Loss: {total_loss:.4f}, MIL: {loss_MIL:.4f}")
            optimizer_MT.zero_grad()
            total_loss.backward()
            optimizer_MT.step()


def test(dataloader, model_MT, gt):
    with torch.no_grad():
        model_MT.eval()
        pred = torch.zeros(0).cuda()
        for i, (f_f) in tqdm(enumerate(dataloader)):
            f_f = f_f.cuda()
            logits = model_MT(f_f, seq_len=None)
            logits = torch.mean(logits, 0)
            pred = torch.cat((pred, logits))
        pred = list(pred.cpu().detach().numpy())
        precision, recall, th = precision_recall_curve(list(gt), np.repeat(pred, 16))
        ap = auc(recall, precision)
        return ap
