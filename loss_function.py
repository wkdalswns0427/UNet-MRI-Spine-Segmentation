import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.nn.functional import one_hot as __functional_one_hot

_default_reduction = 'mean'
_epsilon = 1e-7


def _apply_reduction(tensor, reduction):
    if reduction is None:
        return tensor
    elif reduction == 'mean':
        return tensor.mean()
    elif reduction == 'sum':
        return tensor.sum()
    raise ValueError("Reduction expected to be None, `mean`, or `sum`, got `%s`" % reduction)


def _dice_loss(mul, add, nd, reduction=_default_reduction, epsilon=_epsilon):
    intersection = mul.sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    union = add.sum(dim=tuple(range(-nd, 0, -1))) + epsilon * 2
    loss = 1. - (2. * intersection / union)
    return _apply_reduction(loss, reduction)


def _iou_loss(mul, add, nd, reduction=_default_reduction, epsilon=_epsilon):
    intersection = mul.sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    union = (add - mul).sum(dim=tuple(range(-nd, 0, -1))) + epsilon
    loss = 1. - (intersection / union)
    return _apply_reduction(loss, reduction)


def dice_loss_2d(output, target, reduction=_default_reduction):
    target = one_hot_nd(target, output.size(-3), 2)
    mul, add = output * target, output + target
    if mul.ndim == 3:
        mul = mul.unsqueeze(0)
    if add.ndim == 3:
        add = add.unsqueeze(0)
    mul = mul[:, :mul.size(-3) - 1, :, :]
    add = add[:, :add.size(-3) - 1, :, :]
    return _dice_loss(mul, add, nd=2, reduction=reduction)


def iou_loss_2d(output, target, reduction=_default_reduction):
    target = one_hot_nd(target, output.size(-3), 2)
    mul, add = output * target, output + target
    if mul.ndim == 3:
        mul = mul.unsqueeze(0)
    if add.ndim == 3:
        add = add.unsqueeze(0)
    mul = mul[:, :mul.size(-3) - 1, :, :]
    add = add[:, :add.size(-3) - 1, :, :]
    return _iou_loss(mul, add, nd=2, reduction=reduction)


def one_hot_nd(tensor, n_classes, nd):  # N H W
    new_shape = list(range(tensor.ndim))
    new_shape.insert(-nd, tensor.ndim)
    return __functional_one_hot(tensor.long(), n_classes).permute(new_shape)  # N C H W


class BCEDiceIoUWithLogitsLoss2d(_WeightedLoss):  # with sigmoid

    def __init__(self, dice_factor=1., bce_factor=2., iou_factor=2., bce_weight=None, reduction='mean'):
        super().__init__(weight=bce_weight, reduction=reduction)
        self.bce_factor = bce_factor
        self.dice_factor = dice_factor
        self.iou_factor = iou_factor

    def forward(self, logit, target):
        bce = F.binary_cross_entropy_with_logits(logit, target, weight=self.weight, reduction=self.reduction) * \
            self.bce_factor
        if not self.training:
            with torch.no_grad():
                probability = one_hot_nd(logit.argmax(-3), logit.size(-3), 2)
        else:
            probability = logit.softmax(dim=-3)  # logit.sigmoid()
        mul, add = probability * target, probability + target
        if mul.ndim == 3:
            mul = mul.unsqueeze(0)
        if add.ndim == 3:
            add = add.unsqueeze(0)
        mul = mul[:, :mul.size(-3) - 1, :, :]
        add = add[:, :add.size(-3) - 1, :, :]
        dice = _dice_loss(mul, add, nd=2, reduction=self.reduction) * self.dice_factor  # noqa
        iou = _iou_loss(mul, add, nd=2, reduction=self.reduction) * self.iou_factor  # noqa
        loss_all = (bce + iou + dice) / (self.bce_factor + self.dice_factor + self.iou_factor)
        return loss_all
