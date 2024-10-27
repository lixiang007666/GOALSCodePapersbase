import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 2020NIPS, When Does Label Smoothing Help? 对gt label进行一个smoothing的操作
# 只对layer3的gt进行label smoothing
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, alpha=0.05, n_class=4):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.n_class = n_class

    def forward(self, logits, target):
        if len(target.shape) != len(logits.shape): # logits维度batchsize*n_class*h*w
            target = torch.unsqueeze(target, 1) # target维度batchsize*1*h*w
        with torch.no_grad():
            # one_hot_target = torch.zeros(size=(target.size(0), self.n_class), device=target.device).scatter_(1, target.view(-1, 1), 1)
            single_label_list = []

            for c in range(self.n_class):
                single_label = (target == c)
                single_label = torch.squeeze(single_label, 1)
                single_label_list.append(single_label)

            one_hot_target = torch.stack(tuple(single_label_list), axis = 1) # 将label转换为oen-hot，维度：batchsize*numclasses*h*w，使用tuple是因为其不可变，代替list更安全
            uniform_target = torch.ones(size=(target.size(0), self.n_class, target.size(2), target.size(3)), device=target.device) * (1 / self.n_class)
            
            mask = torch.zeros(size=(target.size(0), self.n_class, target.size(2), target.size(3)), device=target.device)
            for i in range(4):
                mask[:,i,:,:] = one_hot_target[:,2,:,:] # 分割第三层的mask
                # mask[:,i,:,:] = one_hot_target[:,1,:,:]

            # label_smoothed_target = one_hot_target * (1 - self.alpha) + uniform_target * self.alpha
            label_smoothed_target = (one_hot_target * (1 - self.alpha) + uniform_target * self.alpha)*mask+one_hot_target*(1-mask) # 只对第三层进行label smoothing

            

        logprobs = F.log_softmax(logits, dim=1)
        loss = - (label_smoothed_target * logprobs).sum(1)
        # return one_hot_target.shape, label_smoothed_target.shape
        # return uniform_target.shape
        return loss.mean()


# 对所有层进行label smoothing
class LabelSmoothingall(nn.Module):
    def __init__(self, alpha=0.05, n_class=4):
        super(LabelSmoothingall, self).__init__()
        self.alpha = alpha
        self.n_class = n_class

    def forward(self, logits, target):
        if len(target.shape) != len(logits.shape): # logits维度batchsize*n_class*h*w
            target = torch.unsqueeze(target, 1) # target维度batchsize*1*h*w
        with torch.no_grad():
            # one_hot_target = torch.zeros(size=(target.size(0), self.n_class), device=target.device).scatter_(1, target.view(-1, 1), 1)
            single_label_list = []

            for c in range(self.n_class):
                single_label = (target == c)
                single_label = torch.squeeze(single_label, 1)
                single_label_list.append(single_label)

            one_hot_target = torch.stack(tuple(single_label_list), axis = 1) # 将label转换为oen-hot，维度：batchsize*numclasses*h*w，使用tuple是因为其不可变，代替list更安全
            uniform_target = torch.ones(size=(target.size(0), self.n_class, target.size(2), target.size(3)), device=target.device) * (1 / self.n_class)
            
            # mask = torch.zeros(size=(target.size(0), self.n_class, target.size(2), target.size(3)), device=target.device)
            # for i in range(4):
            #     mask[:,i,:,:] = one_hot_target[:,2,:,:]
            label_smoothed_target = one_hot_target * (1 - self.alpha) + uniform_target * self.alpha
            # label_smoothed_target = (one_hot_target * (1 - self.alpha) + uniform_target * self.alpha)*mask+one_hot_target*(1-mask) # 只对第三层进行label smoothing

            

        logprobs = F.log_softmax(logits, dim=1)
        loss = - (label_smoothed_target * logprobs).sum(1)
        # return one_hot_target.shape, label_smoothed_target.shape
        # return uniform_target.shape
        return loss.mean()


# class LabelSmoothingCrossEntropyLoss(nn.Module):
#     def __init__(self, alpha=0.1, n_class=4):
#         super(LabelSmoothingCrossEntropyLoss, self).__init__()
#         self.alpha = alpha
#         self.n_class = n_class

#     def forward(self, logits, target):
#         if len(target.shape) != len(logits.shape):
#             target = torch.unsqueeze(target, 1) # labels维度batchsize*1*h*w
#         with torch.no_grad():
#             # one_hot_target = torch.zeros(size=(target.size(0), self.n_class), device=target.device).scatter_(1, target.view(-1, 1), 1)
#             single_label_list = []

#             for c in range(self.n_class):
#                 single_label = (target == c)
#                 single_label = torch.squeeze(single_label, 1)
#                 single_label_list.append(single_label)

#             one_hot_target = torch.stack(tuple(single_label_list), axis = 1) # 将label转换为oen-hot，维度：batchsize*numclasses*h*w，使用tuple是因为其不可变，代替list更安全
#             uniform_target = torch.ones(size=(target.size(0), self.n_class), device=target.device) * (1 / self.n_class)
#             label_smoothed_target = one_hot_target * (1 - self.alpha) + uniform_target * self.alpha

#         logprobs = F.log_softmax(logits, dim=-1)
#         loss = - (label_smoothed_target * logprobs).sum(-1)
#         return loss.mean()


class DiceLoss(nn.Module):
    """
    Implements the dice loss function.
    Args:
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """
    def __init__(self, ignore_index = 3): # 无用像素是其他类别，mask像素值设为3
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.eps = 1e-5 # 防止分母为0加上此参数

    def forward(self, logits, labels):
        if len(labels.shape) != len(logits.shape):
            labels = torch.unsqueeze(labels, 1) # labels维度batchsize*1*h*w
        num_classes = logits.shape[1] # 分割类别
        # mask = (labels != self.ignore_index)
        # mask = labels
        # logits = logits * mask
        single_label_list = []

        for c in range(num_classes):
            single_label = (labels == c)
            single_label = torch.squeeze(single_label, 1)
            single_label_list.append(single_label)
        labels_one_hot = torch.stack(tuple(single_label_list), axis = 1) # 将label转换为oen-hot，维度：batchsize*numclasses*h*w，使用tuple是因为其不可变，代替list更安全
        logits = F.softmax(logits, dim = 1) # logits维度batchsize*4*h*w
        dims = (0,2,3) # 压缩0，2，3这三个维度，最后得到的loss是一个长度为4的一维向量，其值分别为4个类别的dice
        intersection = torch.sum(logits * labels_one_hot, dims)
        cardinality = torch.sum(logits + labels_one_hot, dims)
        dice_score = (2. * intersection / (cardinality + self.eps))
        dice_loss = (1-dice_score).mean()
        return dice_loss, dice_score # 返回四个类别的dice_score



class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor
        gamma:
        ignore_index:
        reduction:
    """

    def __init__(self, num_class, alpha=None, gamma=2, ignore_index=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.ignore_index = ignore_index
        self.alpha = alpha
        if alpha is None:
            self.alpha = torch.ones(num_class, )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError('the length not equal to number of class')

        # if isinstance(self.alpha, (list, tuple, np.ndarray)):
        #     assert len(self.alpha) == self.num_class
        #     self.alpha = torch.Tensor(list(self.alpha))
        # elif isinstance(self.alpha, (float, int)):
        #     assert 0 < self.alpha < 1.0, 'alpha should be in `(0,1)`)'
        #     assert balance_index > -1
        #     alpha = torch.ones((self.num_class))
        #     alpha *= 1 - self.alpha
        #     alpha[balance_index] = self.alpha
        #     self.alpha = alpha
        # elif isinstance(self.alpha, torch.Tensor):
        #     self.alpha = self.alpha
        # else:
        #     raise TypeError('Not support alpha type, expect `int|float|list|tuple|torch.Tensor`')

    def forward(self, logit, target):
        # assert isinstance(self.alpha,torch.Tensor)\
        N, C = logit.shape[:2]
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit, dim=1)
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        ori_shp = target.shape
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]
        valid_mask = None
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            target = target * valid_mask

        # ----------memory saving way--------
        prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        logpt = torch.log(prob)
        # alpha_class = alpha.gather(0, target.view(-1))
        alpha_class = alpha[target.squeeze().long()]
        class_weight = -alpha_class * torch.pow(torch.sub(1.0, prob), self.gamma)
        loss = class_weight * logpt
        if valid_mask is not None:
            loss = loss * valid_mask.squeeze()

        if self.reduction == 'mean':
            loss = loss.mean()
            if valid_mask is not None:
                loss = loss.sum() / valid_mask.sum()
        elif self.reduction == 'none':
            loss = loss.view(ori_shp)
        return loss