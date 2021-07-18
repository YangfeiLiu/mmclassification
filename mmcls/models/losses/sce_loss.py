import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class SCELoss(nn.Module):
    '''
    Symmetric Cross Entropy for Robust Learning with Noisy Labels
    '''
    def __init__(self, alpha=1.0, beta=1.0, use_sigmoid=False):
        '''
        :param alpha:
        :param beta:
        :param use_sigmoid: 是否使用sigmoid激活，在多标签分类中置为True
        '''
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.use_sigmoid = use_sigmoid

    def forward(self, pred, labels, **kwargs):
        # CE
        num_classes = pred.size(1)
        # labels = labels.long().squeeze(dim=1)
        if self.use_sigmoid:
            ce = F.binary_cross_entropy_with_logits(pred, labels, reduction='mean')
        else:
            ce = F.cross_entropy(pred, labels, reduction='mean')

        # RCE
        pred = F.sigmoid(pred) if self.use_sigmoid else F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        labels_one_hot = F.one_hot(labels, num_classes).float().to(self.device)
        labels_one_hot = torch.clamp(labels_one_hot, min=1e-4, max=1.0)

        if self.use_sigmoid:
            rce = -1 * F.binary_cross_entropy(pred, torch.log(labels_one_hot))
        else:
            rce = -1 * F.nll_loss(pred, torch.log(labels_one_hot))

        loss = self.alpha * ce + self.beta * rce
        return loss


