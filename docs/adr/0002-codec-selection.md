# 0002. Neural Codec Selection & Feasibility Contract (Meta EnCodec 24kHz Causal)

During Phase 0 (#9 Codec Feasibility & Streaming Contract), we evaluated candidate streaming neural codecs to anchor the TimbreLite causal conversion pipeline. We selected Meta EnCodec 24kHz Causal (`facebook/encodec_24khz`) as the foundational backbone over StreamCodec2.

## Status

Accepted

## Considered Options

- **Candidate A: StreamCodec2 (arXiv:2509.13670)**:
  - *Specifications*: 16 kHz sample rate, 20 ms causal frame hop (320 samples), MDCT spectral front-end, causal ConvNeXt v2 blocks, RSVQ quantizer, IMDCT synthesis, ~5.4M parameters, 910 MFLOPs.
  - *Evaluation*: As of September 2026, no official open-source model weights or checkpoints are publicly distributed. Furthermore, 16 kHz audio imposes an 8 kHz Nyquist cutoff that degrades vocal sibilance, air, and presence for target persona representation.
- **Candidate B: Meta EnCodec 24kHz Causal (`facebook/encodec_24khz`)**:
  - *Specifications*: 24 kHz sample rate, 13.33 ms causal frame hop (320 samples), strided causal convolutions ($8 \times 5 \times 4 \times 2 = 320$), 2-layer causal LSTM, 128-dimensional continuous latent space.
  - *Evaluation*: Official pretrained weights are open and verified (`encodec_24khz-d7cc33bc.th`). Preserves high-frequency speech up to 12 kHz.

## Empirical Verification Results

1. **Continuous Latent Passthrough (Go/No-Go Gate)**:
   - Direct continuous latent bypass ($z_c \to \text{Decoder} \to y_c$) was benchmarked against the standard Residual Vector Quantizer baseline ($z_c \to \text{RVQ} \to \hat{z}_q \to \text{Decoder} \to y_q$):
     - $\text{SI-SDR}(y_c, x) = 24.54\text{ dB}$ (vs $24.11\text{ dB}$ for $y_q$).
     - $\text{Log-Spectral Distance}(y_c, x) = 2.598\text{ dB}$ (vs $2.927\text{ dB}$ for $y_q$).
     - Relative degradation $\Delta \text{LSD} \le 0.0\text{ dB}$ (Continuous bypass improves fidelity by $+0.43\text{ dB}$ SI-SDR).
   - Identity transformation ($z_c \to A_{identity}(z_c) \to \text{Decoder}$) confirmed zero divergence ($\Delta < 1e-6$).
2. **Strict Zero-Lookahead Causality**:
   - Modifying future audio chunks ($t+1, t+2$) produces mathematically identical ($0.0\text{ difference}$) encoder latents and decoded audio for chunk $t$.
3. **Streaming Memory Stability**:
   - Memory allocation remains completely static across continuous chunk streaming with zero memory leakage.

## Consequences

- The streaming frame contract is locked to **24,000 Hz sample rate**, **320 audio samples per chunk (13.33 ms)**, and **128-dimensional continuous latent space ($C_{latent} = 128$)**.
- Downstream transformation modules (#10 Causal Bottleneck, #11 Content Cleanser, #12 Personalized Adapter) will consume and produce 128-dimensional continuous latents ($C_{latent} = 128 \to C_b = 128 \to C_{latent} = 128$).
- Resolves and supersedes the initial exploratory assumption in ADR-0001 regarding StreamCodec2 checkpointing.
