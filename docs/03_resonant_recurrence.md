# Resonant Block Details

Real-valued implementation of complex recurrence:

[h1_t]   [r cos w  -r sin w] [h1_{t-1}] + W x_t
[h2_t] = [r sin w   r cos w] [h2_{t-1}]

Each pair = damped digital resonator.

Multi-scale extension:
Initialize ω across frequency spectrum.
