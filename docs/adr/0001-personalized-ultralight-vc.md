# 0001. One-to-One Personalized Ultra-Lightweight Real-time Voice Conversion

Running voice conversion alongside competitive FPS games requires near-zero GPU contention, sub-40ms end-to-end latency, and natural speech on short callouts. We decided to build a 1-to-1 personalized system using a frozen causal neural audio codec (StreamCodec2), a speaker-invariant Content Cleanser, an in-graph Prosody Head, and a sub-1M parameter causal TCN+GRU adapter, strictly eliminating external inference models (no FAISS, no RMVPE/CREPE, no ASR, no speaker encoder).

## Status

Accepted

## Considered Options

- **General Any-to-Any Real-time VC (e.g. RVC / Seed-VC)**: Rejected because multi-speaker generalization requires heavy feature extractors (HuBERT/ContentVec), retrieval (FAISS), and vocoders, causing 90~170ms latency and high GPU/VRAM contention during gameplay.
- **Direct Codec Latent Mapping ($Z_{source} \to \text{Adapter} \to Z_{target}$)**: Rejected because codec latents are entangled with source timbre; without explicit content cleansing, an ultra-small adapter leaks the user's voice or collapses into unnatural hybrids under unpaired training.
- **Dedicated Neural Pitch Tracking at Inference (RMVPE / CREPE)**: Rejected because BiGRU-based pitch trackers introduce device hops, buffering delays (10~15ms), and unnecessary GPU overhead during gaming.

## Hypotheses to Validate

- **H1 (Personalization)**: Restricting the system to a fixed target speaker significantly reduces required adapter capacity compared to general VC models.
- **H2 (Disentanglement)**: Removing source-speaker identity from codec latents prior to adaptation allows an ultra-small adapter to focus solely on target timbre.
- **H3 (Compression)**: Across bottleneck dimensions ($256 \to 128 \to 64 \to 32$), a Pareto sweet spot exists that preserves speech quality and target timbre with minimal parameters.
- **H4 (Streaming)**: A fully causal pipeline (causal encoder + causal adapter + causal decoder) maintains natural speech quality under short frame streaming (13.33ms / 320 samples @ 24kHz).
- **H5 (Runtime)**: CPU-side VAD admission, lock-free SPSC queues, dedicated inference threading, CUDA Graph ping-pong execution, pre-roll playback, and state lifecycle management reduce audio deadline misses to zero and eliminate perceptible game frametime stutter.

## Verification Targets

- **Target 1 (Content Cleanser)**: Minimize speaker information while preserving phonetic content. Evaluated via decreasing speaker classifier accuracy and speaker verification leakage, while maintaining word error rate ($WER \approx \text{constant}$).
- **Target 2 (Inference Latency)**: Target inference latency: $<1.5\text{ ms}$ under the defined benchmark environment, evaluated across $P50, P95, P99, P99.9$ distributions with target `Audio Underruns = 0`.
- **Target 3 (GPU Workload Gating)**: VC inference workload $= 0$ during silence (zero CUDA Graph launches). True resource isolation achieved if deployed to secondary GPU / iGPU or CPU fallback ($RTF < 0.5$).
- **Target 4 (Jitter vs Latency Trade-off)**: Evaluate the elasticity buffer trade-off between 2 frames ($26.7\text{ ms}$) and 3 frames ($40\text{ ms}$) to balance round-trip latency against Windows scheduling jitter tolerance.

## Implementation Roadmap

1. **Milestone 1 (Baseline)**: StreamCodec2 frozen encoder/decoder with basic causal bottleneck ($1024 \to 128 \to \text{TCN+GRU} \to 1024$) to validate causal reconstruction pipeline.
2. **Milestone 2 (Content Cleanser)**: Stage 1 multi-speaker adversarial training ($L_{content} + L_{speaker}^{GRL} + L_{invariance}$) to establish speaker-invariant representation.
3. **Milestone 3 (Prosody Head & Target Adaptation)**: Stage 2 adapter training with target reconstruction ($L_{recon}$), prosody distillation ($L_{prosody}$), and timbre similarity ($L_{speaker}$).
4. **Milestone 4 (Session Controller)**: CPU VAD, 80ms pre-roll ring buffer, and three-stage GRU state machine (Active, Hangover freeze, Soft decay).
5. **Milestone 5 (Windows Audio Runtime)**: WASAPI `IAudioClient3` low-period shared mode, MMCSS Pro Audio thread, lock-free SPSC queues, and Ping-Pong CUDA Graph engine.
