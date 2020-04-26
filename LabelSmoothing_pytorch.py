import torch
from torch import nn


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, input, target):
        input = input.log_softmax(-1)
        num_classes = input.size(-1)
        true_dist = torch.zeros_like(input)
        true_dist.fill_(self.smoothing / (num_classes - 1))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * input, -1))
