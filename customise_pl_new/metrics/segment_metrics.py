"""Tools to print the segmentation metrics prettily.
"""
import torch


def pretty_print(CLASSES, accuracy, acc_per_cls, mean_acc, iou_per_cls, miou, is_sparse=False, log_func=None):
    """
    Print the segmentation metrics prettily.
    
    Args:
        CLASSES (list): a list of class names, in order
        accuracy (float): the overall accuracy (Pixel Acc)
        acc_per_cls (list): a list of accuracy for each class 
        mean_acc (float): the mean accuracy among all classes (Mean Acc)
        iou_per_cls (list): a list of Intersection over Union (IoU) for each class
        miou (float): the mean Intersection over Union (mIoU) among all classes
        is_sparse (bool, optional): whether the labels are sparse or not, default False
        log_func (function, optional): a function to log the metrics, default None, only support pytorch-lightning logger
    
    Returns:
        None
    """
    
    if log_func is None:
        pass
    num_classes = len(CLASSES)
    for i in range(num_classes):
        log_func(CLASSES[i] + "_acc", acc_per_cls[i])
    for i in range(num_classes):
        log_func(CLASSES[i] + "_iou", iou_per_cls[i])

    log_func("Pixel Acc", accuracy)
    log_func("Mean Acc", mean_acc)
    log_func("mIoU", miou)


def nanmean(v, *args, inplace=True, **kwargs):
    """calculate the mean of v that contains nan values.
    """
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


class SegmentEvaluator:
    """Calculate the Pixel Acc and Mean Acc, and other metrics based on segmentation masks.
    """
    def __init__(self, is_sparse=False, categories=None):
        self.is_sparse = is_sparse
        self.categories = categories

    def __call__(self, confmat, log_func, pre_fix="train"):
        if self.is_sparse:
            true_confmat = confmat[:-1, :-1]
        else:
            true_confmat = confmat

        # calculate pixel accuracy
        with torch.no_grad():
            correct = torch.diag(true_confmat).sum()
            total = true_confmat.sum()

            # pixel acc and mean class accuracy
            accuracy = correct / total
            acc_per_cls = (torch.diag(true_confmat) / true_confmat.sum(axis=1))
            mean_acc = nanmean(acc_per_cls)

            # iou
            intersection = torch.diag(true_confmat)
            union = true_confmat.sum(0) + true_confmat.sum(1) - intersection
            iou_per_cls = intersection.float() / union.float()
            iou_per_cls[torch.isinf(iou_per_cls)] = float('nan')
            miou = nanmean(iou_per_cls)

        if log_func is not None:
            log_func(pre_fix + "_acc", accuracy.tolist(), prog_bar=True)
            log_func(pre_fix + "_mean_acc", mean_acc.tolist(), prog_bar=True)
            log_func(pre_fix + "_miou", miou.tolist(), prog_bar=True)

            # log per category performance
            if self.categories is not None:
                for idx, acc in enumerate(acc_per_cls.cpu().numpy()):
                    log_func(pre_fix + "_acc_cat_" + self.categories[idx], acc)
        return accuracy.tolist(), acc_per_cls.tolist(), mean_acc.tolist(), iou_per_cls.tolist(), miou.tolist()
