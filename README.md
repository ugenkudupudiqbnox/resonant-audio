# Resonant Linear Recurrent Networks (ReLRN)

## Overview
ReLRN explores oscillatory recurrent dynamics for streaming audio modeling.

Instead of modeling memory as pure exponential decay,
we model memory as **learnable damped resonators**.

This repository is structured as a research-first architecture lab.

---

## Core Recurrence

Standard LRNN:
h_t = α ∘ h_{t-1} + W x_t

Resonant LRNN:
h_t = r · R(ω) h_{t-1} + W x_t

Each 2D hidden pair acts as a digital resonator.

---

## Research Focus
- Streaming-native modeling
- Oscillatory state dynamics
- Interpretable resonance spectrum
- Hardware-efficient recurrence

Created: 2026-02-14T21:10:06.851257 UTC
