import torch
import torch.nn as nn
import torch.optim as optim
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

    for _ in tqdm(range(100)):  # 100 batches per epoch
        x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        preds, _ = model(x)

        loss = criterion(preds, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()

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
