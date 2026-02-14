import torch
import torch.nn as nn
from .resonant_block import ResonantBlock


class ResonantAudioModel(nn.Module):
    """
    Stacked Resonant LRNN for spectrogram modeling.
    """

    def __init__(
        self,
        input_dim=80,
        hidden_dim=256,
        num_layers=8,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [ResonantBlock(hidden_dim) for _ in range(num_layers)]
        )

        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def init_state(self, batch_size, device):
        return [
            torch.zeros(batch_size, block.dim, device=device)
            for block in self.blocks
        ]

    def forward(self, x, state=None):
        """
        x: (B, T, input_dim)
        """

        B, T, _ = x.shape

        if state is None:
            state = self.init_state(B, x.device)

        outputs = []
        x = self.input_proj(x)

        for t in range(T):
            xt = x[:, t]

            new_state = []

            for i, block in enumerate(self.blocks):
                xt, h_new = block(xt, state[i])
                new_state.append(h_new)

            state = new_state
            outputs.append(xt.unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        y = self.output_proj(y)

        return y, state
