# Resonant Linear Recurrent Networks for Streaming Audio Modeling

## Overview
Resonant LRNNs model audio as a bank of learnable damped resonators.
Designed for streaming, constant-memory inference, and architectural novelty.

## Core Idea
Standard LRNN:
h_t = alpha * h_{t-1} + W x_t

Resonant LRNN:
h_t = r * R(omega) * h_{t-1} + W x_t

Each hidden pair becomes a digital resonator.

## Goals
- Streaming-native audio modeling
- Oscillatory memory representation
- Interpretable resonance spectrum
- Parameter-efficient architecture
