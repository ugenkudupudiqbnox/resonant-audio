# Basic shape test

import torch
from src.models.resonant_lrnn import ResonantBlock

def test_resonant_shape():
    model = ResonantBlock(256)
    x = torch.randn(4, 256)
    h = torch.zeros(4, 256)
    out, _ = model(x, h)
    assert out.shape == (4, 256)
