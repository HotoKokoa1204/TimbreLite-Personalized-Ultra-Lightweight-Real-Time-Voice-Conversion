# 0004. Explicit F0 Conditioning via Statistical Normalization

We decided to decouple pitch control from neural latent representation guessing by introducing explicit F0 conditioning with statistical distribution normalization (median/IQR alignment) and FiLM modulation, strictly avoiding waveform-level pitch shifting and neural F0 prediction models.

## Status

Accepted

## Context & Problem Statement

Initial end-to-end voice conversion experiments (Epoch 82) produced a low-pitched, metallic "male robotic" voice when converting from the male Source Speaker (median F0 ~131 Hz) to Hu Tao (median F0 ~393 Hz). This was caused by three architectural bottlenecks:
1. The implicit prosody head was forced to predict pitch without supervision, retaining the male fundamental frequency register.
2. The global `skip_proj` shortcut leaked low-frequency male harmonics directly into the decoder.
3. The lack of paired training data rendered cross-speaker neural F0 prediction ill-posed and unstable.

## Considered Options

- **Candidate A: Neural F0 Predictor ($F0_{src} \to \text{Net} \to F0_{tgt}$)**: Rejected because without paired parallel speech, predicting target pitch from source pitch introduces high variance, hallucination, and unnatural intonation.
- **Candidate B: Waveform-Level Pitch Shifting (+18.9 semitones pre-processing)**: Rejected because pitch-shifting the raw waveform before the codec drastically shifts formants, producing chipmunk distortion and phase artifacts.
- **Candidate C: Explicit F0 Extraction with Robust Statistical Normalization & FiLM Modulation**: Accepted.

## Decision & Specifications

1. **Deterministic Target F0 Normalization**:
   We map source pitch to target register using robust non-parametric statistics:
   $$\log F0_{out} = \text{med}_t + \frac{\text{IQR}_t}{\text{IQR}_s} (\log F0_{src} - \text{med}_s)$$
   with optional user pitch bias $\Delta k$ (default 0 semitones):
   $$\log F0_{final} = \log F0_{out} + \Delta k \frac{\ln 2}{12}$$
   Production statistics ($\text{med}_s, \text{IQR}_s, \text{med}_t, \text{IQR}_t$) are strictly derived from the training splits only, reserving validation sets for evaluation.

2. **Asymmetric Tracker Parity Contract**:
   - **Offline Target (Hu Tao)**: High-quality pitch extraction across 441 train utterances ($f \in [140, 800]\text{ Hz}$).
   - **Offline Source (User)**: Exact replay of the runtime causal pitch tracker across 260 train utterances ($f \in [65, 380]\text{ Hz}$), guaranteeing $\text{med}_s / \text{IQR}_s$ perfectly match real-time inference characteristics and eliminating tracker bias.
   - **Runtime Tracker**: Pure CPU causal autocorrelation/YIN operating on a 1280-sample causal buffer (53.33ms = 4 chunks) with 320-sample hop (13.33ms) and lookahead = 0.

3. **Voicing & Boundary Invariants**:
   - Hysteresis voicing decision: Voiced ON at periodicity $\ge 0.40$, Voiced OFF at periodicity $\le 0.30$, holding previous state between $[0.30, 0.40]$. Energy gate forces silence to unvoiced.
   - Interpolation: Short unvoiced gaps ($\le 2$ frames / $26.7\text{ms}$) are linearly interpolated; long gaps ($> 2$ frames) are zeroed.
   - Dynamic Delta: $\Delta\log F0$ is computed only across Voiced-to-Voiced transitions; boundaries involving unvoiced frames yield $\Delta = 0$, clamped to $[-0.5, +0.5]$.

4. **FiLM Modulation Conditioning**:
   The 3D vector $[\text{normalized\_logF0}, \Delta\text{logF0}, \text{VUV}]$ is projected to modulate intermediate content representations via Feature-wise Linear Modulation:
   $$H' = \gamma(F0) \odot H + \beta(F0)$$

## Consequences

- Completely eliminates pitch ambiguity from the content cleanser and adapter.
- Isolates pitch conversion ($131\text{ Hz} \to 393\text{ Hz}$) from timbre/formant conversion.
- Enables EXP-1A statistical verification before retrained model ablation.
