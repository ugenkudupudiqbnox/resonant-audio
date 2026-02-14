import torch
import torch.nn as nn

class LRNNBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.decay = nn.Parameter(torch.zeros(dim))
        self.input_proj = nn.Linear(dim, dim)

    def forward(self, x, h):
        alpha = torch.sigmoid(self.decay)
        h = alpha * h + self.input_proj(x)
        return h, h
