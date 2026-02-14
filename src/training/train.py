import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm

from src.models.resonant_model import ResonantAudioModel


def dummy_dataset(batch_size=16, seq_len=200, input_dim=80):
    """
    Temporary synthetic dataset.
    Replace later with real mel spectrogram dataset.
    """
    while True:
        x = torch.randn(batch_size, seq_len, input_dim)
        y = x.clone()  # identity task for debugging
        yield x, y


def train_one_epoch(model, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    data_iter = dummy_dataset()
    
    # Tracking metrics
    num_layers = len(model.blocks)
    h_norms = [0.0] * num_layers
    grad_norms = [0.0] * num_layers

    for _ in tqdm(range(100)):  # 100 batches per epoch
        x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        preds, final_state = model(x)

        loss = criterion(preds, y)

        loss.backward()

        # Track hidden state norms and gradient norms
        with torch.no_grad():
            for i, block in enumerate(model.blocks):
                h_norms[i] += final_state[i].norm().item()
                
                # Calculate grad norm for this block
                block_grad_norm = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        block_grad_norm += p.grad.detach().data.norm(2).item() ** 2
                grad_norms[i] += block_grad_norm ** 0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

    # Log advanced stats across layers
    with torch.no_grad():
        print("\n--- Layer-wise Statistics ---")
        for i, block in enumerate(model.blocks):
            r = torch.sigmoid(block.r_param)
            w = math.pi * torch.tanh(block.w_param)
            
            avg_h_norm = h_norms[i] / 100
            avg_grad_norm = grad_norms[i] / 100
            
            print(f"Layer {i} | Mean r: {r.mean().item():.4f} | Mean |w|: {w.abs().mean().item():.4f}")
            print(f"        | Avg H-Norm: {avg_h_norm:.4f} | Avg Grad-Norm: {avg_grad_norm:.4f}")

    return total_loss / 100


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResonantAudioModel(
        input_dim=80,
        hidden_dim=256,
        num_layers=4,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.L1Loss()

    epochs = 5

    for epoch in range(epochs):
        loss = train_one_epoch(model, optimizer, criterion, device)

        print(f"Epoch {epoch+1} | Loss: {loss:.6f}")

    print("Training complete.")


if __name__ == "__main__":
    main()
