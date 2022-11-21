import typing

import numpy as np
import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature, cuda, *args, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cuda = cuda
        
    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, out_1, out_2):
        batch_size = out_1.shape[0]
        
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = self.get_negative_mask(batch_size)
        if self.cuda:
            mask = mask.cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        Ng = neg.sum(dim=-1)

        # contrastive loss
        loss = (-torch.log(pos / (pos + Ng))).mean()
        
        return loss
    
    
class DebiasedNegLoss(nn.Module):
    def __init__(self, temperature, cuda, tau_plus):
        super(DebiasedNegLoss, self).__init__()
        self.temperature = temperature
        self.tau_plus = tau_plus
        self.cuda = cuda
        
    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, out_1, out_2):
        batch_size = out_1.shape[0]
        
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = self.get_negative_mask(batch_size)
        if self.cuda:
            mask = mask.cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        N = batch_size * 2 - 2
        Ng = (-self.tau_plus * N * pos + neg.sum(dim=-1)) / (1 - self.tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))

        # contrastive loss
        loss = (-torch.log(pos / (pos + Ng))).mean()
        
        return loss
    
    
class DebiasedPosLoss(nn.Module):
    def __init__(self, temperature, cuda, tau_plus):
        super(DebiasedPosLoss, self).__init__()
        self.temperature = temperature
        self.tau_plus = 1 - tau_plus
        self.cuda = cuda
        
    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)
        return negative_mask

    def forward(self, out_1, out_2):
        batch_size = out_1.shape[0]
        
        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)
        mask = self.get_negative_mask(batch_size)
        if self.cuda:
            mask = mask.cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        N = batch_size * 2 - 2
        Ng = (N * self.tau_plus - (1 - self.tau_plus)) * neg.mean(dim=-1)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        
        p = (1 - self.tau_plus) * neg.mean(dim=-1)
        # constrain (optional)
        p = torch.clamp(pos - p, min=np.e ** (-1 / self.temperature))

        # contrastive loss
        loss = (-torch.log(p / (pos + Ng))).mean()
        
        return loss
    
    
def get_loss(name: str) -> typing.Union[typing.Type[ContrastiveLoss], typing.Type[DebiasedNegLoss], typing.Type[DebiasedPosLoss]]:
    if name == "Contrastive":
        return ContrastiveLoss
    if name == "DebiasedNeg":
        return DebiasedNegLoss
    if name == "DebiasedPos":
        return DebiasedPosLoss
    raise Exception("Unknown dataset {}".format(name))
