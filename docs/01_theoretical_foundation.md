# Theoretical Foundation

## 1. Exponential Memory

Traditional recurrence:
h_t = α h_{t-1} + W x_t

Expands to:
h_t = Σ α^k W x_{t-k}

Pure decay.

## 2. Resonant Memory

Resonant recurrence:
h_t = r R(ω) h_{t-1} + W x_t

Expands to:
h_t = Σ r^k e^{ikω} W x_{t-k}

This creates a damped sinusoidal kernel.

## 3. Stability

System stable if:
r < 1

Parameterization:
r = sigmoid(a)
ω = π tanh(b)

Ensures bounded eigenvalues.
