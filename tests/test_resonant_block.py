import torch
from src.models.resonant_block import ResonantBlock


def test_shape_consistency():
    block = ResonantBlock(dim=128)
    x = torch.randn(4, 128)
    h = torch.zeros(4, 128)

    out, h_new = block(x, h)

    assert out.shape == (4, 128)
    assert h_new.shape == (4, 128)


def test_stability_constraint():
    block = ResonantBlock(dim=128)

    r = torch.sigmoid(block.r_param)

    assert torch.all(r < 1.0)
    assert torch.all(r > 0.0)


def test_zero_input_decay():
    block = ResonantBlock(dim=128)
    x = torch.zeros(2, 128)
    h = torch.randn(2, 128)

    _, h_new = block(x, h)

    # Norm should not explode
    assert torch.norm(h_new) <= torch.norm(h) * 1.1
