"""Implement all loss terms for material segmentation.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class CrossEntropyLoss2d(nn.Module):
    """Implementation of the cross entropy loss
    """
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        """Calaulate cross entropy loss.

        Args:
            inputs: predicted segmentation, without softmax
            targets: ground truth segmentation

        Returns: the cross entropy loss between inputs and targets
        """
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


def one_hot(index, classes):
    """
    Converts an index to a one-hot vector.
    
    Args:
        index (torch.Tensor): Tensor containing the original index.
        classes (int): Number of classes.
    
    Returns:
        torch.Tensor: The transformed one-hot vector.
    
    """
    # index is flatten (during ignore) ##################
    
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.Tensor(size).fill_(0)
    mask = mask.type_as(index)
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):
    """Focal loss implementation, from https://github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, gamma=0, eps=1e-7, size_average=True, one_hot=True, ignore=-1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        
        self.size_average = size_average
        self.one_hot = one_hot
        self.ignore = ignore

    def tensor_forward(self, input, target, softmax=False):
        """
        Args:
            input (torch.Tensor): Predicted tensor with shape (B, C, H, W).
            target (torch.Tensor): Ground truth tensor with shape (B, C, H, W).
            softmax (bool, optional): Whether to perform softmax operation on the output. The default is False.
        
        Returns:
            focal loss and number of labelled samples.
        
        """
        B, C, H, W = input.size()
        input = input.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
        target = target.view(-1)
        if self.ignore is not None:
            valid = (target != self.ignore)
            input = input[valid]
            target = target[valid]

        if self.one_hot: target = one_hot(target, input.size(1))
        if softmax:
            probs = F.softmax(input, dim=1)
        else:
            probs = input
        probs = (probs * target).sum(1)
        probs = probs.clamp(self.eps, 1. - self.eps)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -(torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        count = len(batch_loss)

        return loss, count

    def forward(self, input, target, softmax=False):
        """
        Forward the focal loss, which supports the use of list that contains tensors with 
        varying shapes.
        
        Args:
            input (torch.Tensor): Predicted tensor with shape (B, C, H, W).
            target (torch.Tensor): Ground truth tensor with shape (B, C, H, W).
            softmax (bool, optional): Whether to perform softmax operation on the output. The default is False.
        
        Returns:
            focal loss and number of labelled samples.
        
        """
        is_list = isinstance(input, list)
        if not is_list:
            return self.tensor_forward(input, target, softmax)
        else:
            loss, count = [], []
            for idx in range(len(input)):
                loss_tmp, count_tmp = self.tensor_forward(input[idx], target[idx], softmax)
                loss.append(loss_tmp)
                count.append(count_tmp)
            count = torch.tensor(count)
            loss = torch.tensor(loss) * count
            loss = loss.sum() / count.sum()

            return loss, count.sum()


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5, ignore=-1):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.ignore = ignore

    def tensor_forward(self, pred, gt, softmax):
        n, c, _, _ = pred.shape
        # softmax so that predicted map can be distributed in [0, 1]
        if softmax:
            pred = torch.softmax(pred, dim=1)

        # one-hot vector of ground truth
        gt[gt == self.ignore] = c
        one_hot_gt = F.one_hot(gt, c + 1)
        one_hot_gt = one_hot_gt[:, :, :, :c].permute(0, 3, 1, 2).contiguous().double()

        # boundary map
        gt_b = F.max_pool2d(
            1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss, 1

    def forward(self, pred, gt, softmax=False):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """
        is_list = isinstance(pred, list)
        if not is_list:
            return self.tensor_forward(pred, gt, softmax)

        else:
            loss, count = [], []
            for idx in range(len(pred)):
                loss_tmp, count_tmp = self.tensor_forward(pred[idx], gt[idx], softmax)
                loss.append(loss_tmp)
                count.append(count_tmp)
            count = torch.tensor(count)
            loss = torch.tensor(loss) * count
            loss = loss.sum() / count.sum()

            return loss, count.sum()

    def one_hot(self, label, n_classes, requires_grad=False):
        """Return One Hot Label"""
        device = label.device
        one_hot_label = torch.eye(
            n_classes, device=device, requires_grad=requires_grad)[label]
        one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

        return one_hot_label
