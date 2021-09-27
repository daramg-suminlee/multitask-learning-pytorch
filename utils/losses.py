
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()

    def forward(self, yhat_list: list, y_list: list):
        loss = 0
        for yhat, y in zip(yhat_list, y_list):
            loss += F.mse_loss(yhat, y.view(-1,1))
        return loss
