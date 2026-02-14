import torch
import torch.nn as nn
import math

class ResonantBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        self.r_param = nn.Parameter(torch.zeros(dim // 2))
        self.w_param = nn.Parameter(torch.zeros(dim // 2))
        self.input_proj = nn.Linear(dim, dim)

    def forward(self, x, h):
        r = torch.sigmoid(self.r_param)
        w = math.pi * torch.tanh(self.w_param)

        h = h.view(h.size(0), -1, 2)
        h1, h2 = h[..., 0], h[..., 1]

        cos_w = torch.cos(w)
        sin_w = torch.sin(w)

        new_h1 = r * (cos_w * h1 - sin_w * h2)
        new_h2 = r * (sin_w * h1 + cos_w * h2)

        h = torch.stack([new_h1, new_h2], dim=-1).view(h.size(0), -1)

        h = h + self.input_proj(x)
        return h, h
