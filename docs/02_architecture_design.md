# Architecture Design

Pipeline:
Waveform → STFT → Log-Mel → Resonant Blocks → Projection → Vocoder

Default Hyperparameters:
- Hidden dimension: 256
- Layers: 8
- Mel bins: 80
- Dropout: 0.1

Stacking:
Input → ResonantBlock × N → Output Projection

Designed for:
- Streaming inference
- O(Td) compute
- O(d) memory
