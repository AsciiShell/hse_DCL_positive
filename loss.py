import typing

import numpy as np
import torch
import torch.nn as nn


def get_negative_mask(batch_size):
    """
    Дла каждого изображения отмечается исходная картинка и ее аугментация
    Получается симметричная матрица с нулевой главной диагональю и 2мя параллельными
    """
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def get_target_mask(target: torch.Tensor) -> torch.Tensor:
    """
    Generate bool matrix for equal targets
    """
    target_ = torch.cat([target, target], dim=0).view(-1, 1)
    mask = (target_ == target_.t().contiguous()) & (target_ != -1)
    return mask


class ContrastiveLossBase(nn.Module):
    def __init__(self, temperature, cuda, *args, **kwargs):
        super().__init__()
        self.temperature = temperature
        self.cuda = cuda

    def forward(self, out_1, out_2, *args):
        batch_size = out_1.shape[0]

        # neg score
        out = torch.cat([out_1, out_2], dim=0)  # shape (2 * bs, fdim)
        # скалярное произведение всех пар
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / self.temperature)  # shape (2 * bs, 2 * bs)
        # 1ая часть (2 * bs, bs) соответствует одной картинке, 2ая часть (2 * bs, bs+1 - 2 * bs) - другой
        mask = self.get_negative_mask(batch_size)  # shape (2 * bs, 2 * bs)
        if self.cuda:
            mask = mask.cuda()
        # оставляем только негативные примеры
        neg = neg.masked_select(mask).view(2 * batch_size, -1)  # shape (2 * bs, 2 * bs - 2)

        # pos score
        # скалярное произведение 2х аугментаций одной картинки
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)  # shape (bs)
        # 1ая часть bs соответствует одной картинке, 2ая часть bs - другой
        pos = torch.cat([pos, pos], dim=0)  # shape (2 * bs)

        return pos, neg


class ContrastiveLoss(ContrastiveLossBase):
    def forward(self, out_1, out_2, target):
        pos, neg = super()(out_1, out_2, target)
        # estimator g()
        Ng = neg.sum(dim=-1)  # shape (2 * bs)
        # contrastive loss
        loss = (-torch.log(pos / (pos + Ng))).mean()
        return loss


class DebiasedNegLoss(ContrastiveLossBase):
    def __init__(self, temperature, cuda, tau_plus):
        super().__init__(temperature, cuda)
        self.tau_plus = tau_plus

    def forward(self, out_1, out_2, out_m):
        pos, neg = super()(out_1, out_2)
        batch_size = out_1.shape[0]

        pos_m = [pos]
        for vec in out_m:
            pos_1 = torch.exp(torch.sum(out_1 * vec, dim=-1) / self.temperature)
            pos_2 = torch.exp(torch.sum(out_2 * vec, dim=-1) / self.temperature)
            pos_new = torch.cat([pos_1, pos_2], dim=0)
            pos_m.append(pos_new)
        pos_m = torch.stack(pos_m, dim=0).mean(dim=0)

        # estimator g()
        N = batch_size * 2 - 2
        Ng = (-self.tau_plus * N * pos_m + neg.sum(dim=-1)) / (1 - self.tau_plus)
        # constrain (optional)
        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / self.temperature))
        # contrastive loss
        loss = (-torch.log(pos / (pos + Ng))).mean()
        return loss


class DebiasedPosLoss(ContrastiveLossBase):
    def __init__(self, temperature, cuda, tau_plus):
        super().__init__(temperature, cuda)
        self.tau_plus = tau_plus

    def forward(self, out_1, out_2, *args):
        pos, neg = super()(out_1, out_2)
        batch_size = out_1.shape[0]

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


def get_loss(name: str) -> typing.Union[typing.Type[ContrastiveLoss],
                                        typing.Type[DebiasedNegLoss],
                                        typing.Type[DebiasedPosLoss]]:
    if name == "Contrastive":
        return ContrastiveLoss
    if name == "DebiasedNeg":
        return DebiasedNegLoss
    if name == "DebiasedPos":
        return DebiasedPosLoss
    raise Exception("Unknown loss {}".format(name))
