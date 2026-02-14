# Vision

Audio signals are oscillatory and harmonic.

Transformers capture global relationships but are not streaming-native.
Standard RNNs model decay but ignore phase continuity.

ReLRN introduces resonance-based recurrence:
Memory as damped oscillation rather than simple forgetting.

Goal:
Demonstrate that oscillatory recurrence provides:
- Better harmonic continuity
- Stronger inductive bias for audio
- Efficient constant-memory inference
