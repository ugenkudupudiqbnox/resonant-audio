# Architecture Design

Pipeline:
Waveform -> STFT -> Log-Mel -> Resonant LRNN Stack -> Vocoder

Key Block:
- Resonant state update
- Optional gating
- Residual connection
