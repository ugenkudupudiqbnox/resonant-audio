import torch
import torch.nn as nn
import math


class ResonantBlock(nn.Module):
    """
    Resonant Linear Recurrent Block

    Each pair of hidden dimensions forms a damped oscillator:

        h_t = r * R(ω) h_{t-1} + W x_t

    Where:
        r = sigmoid(a)
        ω = π * tanh(b)
    """

    def __init__(self, dim, input_dim=None):
        super().__init__()

        assert dim % 2 == 0, "Hidden dimension must be even."

        self.dim = dim
        self.pairs = dim // 2

        input_dim = input_dim or dim

        # Resonance parameters
        self.r_param = nn.Parameter(torch.zeros(self.pairs))
        self.w_param = nn.Parameter(torch.zeros(self.pairs))

        # Input projection
        self.input_proj = nn.Linear(input_dim, dim)

        # Output projection
        self.output_proj = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, h):
        """
        x: (B, D)
        h: (B, D)
        """

        B = x.size(0)

        # Stability-constrained parameters
        r = torch.sigmoid(self.r_param)                     # (pairs,)
        w = math.pi * torch.tanh(self.w_param)              # (pairs,)

        cos_w = torch.cos(w)
        sin_w = torch.sin(w)

        # Reshape hidden state into oscillator pairs
        h = h.view(B, self.pairs, 2)

        h1 = h[:, :, 0]
        h2 = h[:, :, 1]

        # Resonant update
        new_h1 = r * (cos_w * h1 - sin_w * h2)
        new_h2 = r * (sin_w * h1 + cos_w * h2)

        h_new = torch.stack([new_h1, new_h2], dim=-1)
        h_new = h_new.view(B, self.dim)

        # Inject input
        h_new = h_new + self.input_proj(x)

        # Residual output
        out = x + self.output_proj(self.norm(h_new))

        return out, h_new
