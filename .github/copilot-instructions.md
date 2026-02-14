# Resonant Audio (ReLRN) Copilot Instructions

## Project Context
This project implements Resonant Linear Recurrent Networks (ReLRN) for streaming audio modeling. It replaces standard exponential decay in RNNs with learnable damped resonators.

## Architecture & Patterns
- **Oscillator pairs**: The hidden state dimension `dim` must ALWAYS be even. Each pair of dimensions $(h_1, h_2)$ represents a 2D resonator.
- **Stability constraints**:
  - Decay factor $r = \text{sigmoid}(\theta_r)$ ensures $r < 1$ (stability).
  - Angular frequency $\omega = \pi \cdot \text{tanh}(\theta_\omega)$ ensures $\omega \in (-\pi, \pi)$ (Nyquist limit).
- **Recurrence**: Standard forward pass in `ResonantBlock` handles a single time step $(x_t, h_{t-1}) \rightarrow (x_{t+1}, h_t)$.
- **Streaming Native**: Architecture is designed for $O(1)$ state per step. Avoid operations that require full sequence knowledge during the recurrent update.
- **Normalization**: `ResonantBlock` uses `LayerNorm` after the recurrent update but before the residual addition.

## File Structure
- [src/models/](src/models/): Core ReLRN logic.
  - [src/models/resonant_block.py](src/models/resonant_block.py): Primary implementation of the resonant recurrence logic.
  - [src/models/resonant_model.py](src/models/resonant_model.py): High-level multi-layer model.
- [docs/](docs/): Theoretical background and failure modes (e.g., [docs/06_failure_modes.md](docs/06_failure_modes.md)).
- [experiments/](experiments/): YAML configurations for training runs.

## Key Workflows
- **Testing**: Use `pytest` to run tests in [tests/](tests/). All tests must pass before merging changes.
- **Training**: Run [src/training/train.py](src/training/train.py). Note that it currently uses a dummy dataset for development.
- **Hyperparameters**: Managed via YAML files in [experiments/](experiments/). Avoid hardcoding hyperparameters in model files.

## Coding Standards
- **Tensor Shapes**: Always document shapes in comments (e.g., `(B, T, D)` for Batch, Time, Dimension).
- **Initialization**: Recurrent states should be initialized using `model.init_state(batch_size, device)`.
- **Residuals**: `ResonantBlock` uses a residual connection: `out = x + output_proj(norm(h_new))`.
- **Typing**: Use `torch.Tensor` for tensor arguments.
