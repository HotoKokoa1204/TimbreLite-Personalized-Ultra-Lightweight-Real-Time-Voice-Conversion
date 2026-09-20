# 0003. Offline ContentVec Teacher Knowledge Distillation for 1-to-1 Voice Conversion

We decided to use an off-the-shelf ContentVec model strictly offline as a phonetic teacher to distill speaker-invariant representations into an ultra-lightweight causal Content Cleanser ($128 \to 64$), completely excluding ContentVec, ASR models, and neural pitch estimators from the runtime inference graph.

## Status

Accepted

## Context

TimbreLite targets real-time voice conversion during competitive FPS gaming with sub-40ms latency and 0% GPU idle contention. General voice conversion frameworks (such as RVC or Seed-VC) execute heavy self-supervised speech representation encoders (HuBERT / ContentVec, ~94M parameters) and external BiGRU pitch trackers (RMVPE / CREPE) online for every audio frame.

In our 1-to-1 personalized setting (converting the single Source Speaker's voice into the single Target Persona, Hu Tao):
1. **Online Infeasibility**: Running a 94M parameter HuBERT/ContentVec model at runtime consumes prohibitive GPU compute and VRAM bandwidth, introducing 50~100ms algorithmic and buffering latency and causing game framerate stutter.
2. **Offline Teacher Distillation**: Because the Source Speaker is fixed and known (curated from 41.5 minutes of microphone speech), ContentVec can be run strictly offline during dataset preprocessing to generate 768-dimensional phonetic teacher embeddings.
3. **Temporal Sampling Discrepancy**:
   - ContentVec operates at 16 kHz with a 20 ms hop (50 Hz).
   - EnCodec operates at 24 kHz with a 13.33 ms hop (320 samples = 75 Hz).
   - Frame alignment requires mathematical temporal interpolation.
4. **Data Slicing & Training Crop Decoupling**:
   - Preprocessing must slice the raw audio into natural speech utterances bounded strictly between 2.0s and 8.0s on silences $>400\text{ms}$.
   - Fixed-length windowing (e.g. 2.56s or 3.2s) must be deferred to the training dataset DataLoader crop strategy to preserve linguistic prosody and avoid data fragmentation.

## Decision

1. **Strict Offline Boundary**:
   - `lengyue233/content-vec-best` runs strictly offline during data preprocessing.
   - Neither ContentVec nor HuggingFace Transformers shall ever be imported or executed in the live streaming inference graph.
   - Hardware strategy: CUDA GPU batch extraction when available, with automatic CPU fallback.

2. **1D Temporal Linear Alignment**:
   - ContentVec 50 Hz representations ($\tilde{T}_{50} \in \mathbb{R}^{B \times 768 \times T_{50}}$) are resampled along the temporal dimension to 75 Hz ($\tilde{T}_{75} \in \mathbb{R}^{B \times 768 \times T_{75}}$) using 1D linear interpolation (`align_corners=False`), matching the exact EnCodec frame count:
     $$T_{75} = \lfloor N_{24k} / 320 \rfloor$$

3. **Student Projection & Distillation Loss**:
   - A linear student projection head $W_{distill}: 64 \to 768$ (with LayerNorm) maps the causal Content Cleanser output $\hat{c}_t$ to the ContentVec phonetic space: $\hat{T} = W_{distill}(\hat{c}_t)$.
   - The training objective combines cosine similarity and mean squared error:
     $$\mathcal{L}_{distill} = (1 - \text{cosine\_similarity}(\hat{T}, \tilde{T}_{75})) + \lambda_{\text{mse}} \|\hat{T} - \tilde{T}_{75}\|_2^2$$
   - EnCodec 24kHz causal encoder weights remain completely frozen.
   - At inference time, $W_{distill}$ is completely discarded, leaving only the ultra-lightweight causal Content Cleanser ($128 \to 64$).

4. **Dual-Rate Preprocessing Pipeline**:
   - Input: `data/my_voice/錄製.m4a` (41.53 minutes).
   - Processing: PyAV decode -> 48 kHz mono -> 60 Hz 4th-order Butterworth high-pass filter -> transient/click gating -> energy VAD segmentation (2.0s~8.0s) -> loudness normalization to -20 dBFS ($\text{RMS} = 0.1$).
   - Export: 24 kHz WAVs (EnCodec) and 16 kHz WAVs (ContentVec) with 90% train / 10% val manifest indexing.
   - Target Persona: `simon3000/genshin-voice` Chinese Hu Tao archive (521 clean WAVs) downloaded automatically and standardized to -20 dBFS at 24 kHz and 16 kHz.

## Consequences

### Positive
- **Zero Runtime Contention**: Live inference requires zero ContentVec computation, maintaining our $<1.5\text{ms}$ frame inference budget.
- **Phonetic Grounding**: Content Cleanser learns clean, speaker-invariant phonetic representations without requiring unstable adversarial GAN/GRL discriminators or multi-speaker corpora.
- **Flexible Training Crops**: Natural 2-8s utterance segmentation allows training on arbitrary crop durations (2.56s, 3.2s, 5.12s) without re-processing the raw audio.
- **Portability**: Preprocessing runs seamlessly across CPU-only and GPU-accelerated environments.

### Negative
- **Preprocessing Disk Footprint**: Caching dual-rate audio (24kHz and 16kHz) and pre-extracted 768-d float32 teacher tensors requires ~2–4 GB disk storage.
- **Offline Dependency**: Preprocessing requires `transformers` and `av` packages, maintained under optional curation dependencies.
