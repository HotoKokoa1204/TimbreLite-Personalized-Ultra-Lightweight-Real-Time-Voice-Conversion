# Personalized Ultra-Lightweight Voice Conversion (FPS-VC)

A real-time voice conversion system specialized for a single target speaker with minimal GPU and latency footprint during FPS gaming.

## Language

**Source Speaker**:
The fixed human operator speaking into the microphone.
_Avoid_: Arbitrary speaker, input speaker, any speaker

**Target Persona**:
The fixed, unchanging target voice timbre and vocal identity, specifically instantiated as Genshin Impact's Hu Tao (胡桃, Chinese voice edition by Tao Dian).
_Avoid_: Arbitrary target, zero-shot target, reference speaker

**Content Cleanser**:
A causal projection module trained to minimize source-speaker information while preserving phonetic linguistic content.
_Avoid_: Bottleneck encoder, feature extractor, speech tokenizer

**Prosody Head**:
A lightweight causal projection directly predicting prosody and pitch representation from codec latents without an external pitch model during inference.
_Avoid_: Pitch estimator, F0 extractor, RMVPE module

**Personalized Adapter**:
A minimal causal TCN and GRU network dedicated exclusively to mapping speaker-invariant content and prosody into the target timbre manifold.
_Avoid_: Voice converter, diffusion model, retrieval model

**Temporal Timbre State**:
The hidden recurrent state maintaining intra-phrase timbre continuity and preventing frame-level timbre jitter across gaming callouts.
_Avoid_: Smoothing filter, moving average, memory bank

**Session Controller**:
The host-side audio coordinator managing VAD admission, pre-roll playback, and timbre state lifecycle across speech bursts.
_Avoid_: Audio manager, pipeline runner

**Pre-roll**:
A circular audio buffer replayed upon speech onset to preserve initial plosives and consonants from VAD trigger latency.
_Avoid_: Lookahead buffer, future window

**Hangover**:
A brief post-speech duration during which GPU inference is gated but temporal timbre state is frozen to preserve continuity across micro-pauses.
_Avoid_: Hold time, silence delay

**Continuous Latent Space**:
The 128-dimensional continuous embedding per 13.33ms frame produced by the causal neural codec encoder and consumed by the decoder, distinct from discrete multi-codebook indices.
_Avoid_: Discrete tokens, codebook indices, 1024-d latent

**Soft Decay**:
An exponential transition (calculated lazily on host CPU upon speech resumption) that smoothly converges the Temporal Timbre State back to neutral after prolonged silence without GPU workload during idle.
_Avoid_: Hard reset, zeroing out, polling decay
