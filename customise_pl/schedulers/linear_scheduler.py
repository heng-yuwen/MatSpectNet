"""linear schoduler, that increase the learning rate linearly.
"""
from torch.optim.lr_scheduler import LambdaLR

def get_lambda_rule(start_epoch=0, num_epochs=300):
    def lambda_rule(epoch):
        # print(f"linear: {epoch}")
        lr_l = 1.0 - max(0, epoch - start_epoch) / float(num_epochs + 1)
        return lr_l

    return lambda_rule


class LinearLR(LambdaLR):
    def __init__(self, optimizer, start_epoch, num_epochs):
        super(LinearLR, self).__init__(optimizer, lr_lambda=get_lambda_rule(start_epoch, num_epochs))
