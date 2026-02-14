import torch
import torch.nn as nn
import torch.optim as optim
import math
from tqdm import tqdm

from src.models.resonant_model import ResonantAudioModel


def harmonic_dataset(batch_size=16, seq_len=200, input_dim=80):
    while True:
        t = torch.linspace(0, 1, seq_len)
        freq = torch.randint(1, 10, (batch_size, 1)).float()
        signal = torch.sin(2 * torch.pi * freq * t)

        signal = signal.unsqueeze(-1).repeat(1, 1, input_dim)
        noise = 0.05 * torch.randn_like(signal)

        x = signal + noise
        y = signal  # denoise

        yield x, y


def train_one_epoch(model, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    data_iter = harmonic_dataset()
    
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
            print("Omega histogram:", block.w_param.data.mean().item())
            print("R histogram:", torch.sigmoid(block.r_param).mean().item())

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
