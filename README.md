# TimbreLite: Personalized Ultra-Lightweight Real-Time Voice Conversion

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)

**TimbreLite** is a personalized, ultra-lightweight, real-time voice conversion (VC) engine designed for concurrent execution alongside high-refresh gaming and low-latency voice communication workloads.

Traditional voice conversion models require heavy external pitch trackers (e.g., CREPE, RMVPE), large diffusion or transformer backbones (>10M parameters), and significant lookahead latency (>100 ms). TimbreLite eliminates these bottlenecks through an end-to-end causal neural pipeline operating entirely within continuous latent representations with strict sub-250K parameter personal conversion modules, true stateful streaming, and zero-GPU silence gating.

---

## 🚀 Key Architectural Pillars

### 1. True Stateful Causal Audio Codec (Meta EnCodec 24 kHz)
- **Causal Contract**: 24 kHz sampling rate, 320-sample hops (13.33 ms frame duration), $C_{latent} = 128$.
- **Strict Mathematical Streaming Equivalence**: Employs stateful layer-by-layer FIFO convolutional buffers, stateful recurrent LSTM hidden states, and causal overlap-add transposed convolutions.
- **Zero Sliding-Window Recomputation**: Each 13.33 ms chunk executes in $O(1)$ time without recomputing historical frames. Streaming output matches full-utterance batch mode to machine precision ($\max |y_{stream} - y_{batch}| < 1\times 10^{-4}$).
- **Counterfactual Zero-Lookahead**: Verified bit-exact invariance ($\Delta = 0.0$) against future frame perturbations.

### 2. Dual-Stream Conditioning with In-Graph Prosody
- **Speaker-Invariant Content Cleanser**: Casual dilated residual convolutions equipped with Gradient Reversal Layer (GRL) and persona classification adversary to strip source speaker timbre while preserving phonetics.
- **In-Graph Prosody Head**: Extracts pitch ($F_0$) and voicing contours directly from continuous latent space inside the neural execution graph, eliminating external CPU/GPU pitch tracking libraries.
- **Conditioning Fusion**: Flexible Additive and FiLM (Feature-wise Linear Modulation) fusion strategies.

### 3. Sub-250K Personalized Adapter
- **Ultra-Lightweight Target Persona**: Dilated causal temporal convolution blocks (TCN) coupled with a single-layer causal GRU (~107K trainable parameters for standard 64-dim bottleneck, well within the 250K limit).
- **Latency Budget**: $< 1.0$ ms per chunk inference time on standard desktop hardware.

### 4. Zero-GPU Silence Gating & Host Session Controller
- **Two-Stage CPU VAD**: High-speed energy-band filtering and zero-crossing rate computation on host CPU.
- **Micro-Pause Hangover Gating**: Gates GPU inference completely (`IDLE_SKIP`, 0 CUDA launches) during intra-phrase pauses (80–120 ms) while preserving recurrent memory states.
- **Lazy Exponential Soft Decay**: When speech resumes after prolonged silence, applies host-computed scalar decay factor $\alpha = \exp(-\lambda \cdot \Delta t_{silence})$ to the GRU hidden state in $O(1)$ time.
- **Circular Lookback Buffer**: Retains pre-roll chunks to restore plosive consonants ('p', 't', 'k') without clipping onsets.

---

## 📁 Repository Architecture

```
src/timbre_lite/
├── codec/                 # Neural audio codec contracts and streaming engines
│   ├── candidate.py       # Candidate wrappers (EnCodec 24kHz Causal)
│   ├── contract.py        # Codec architectural specification contract
│   ├── metrics.py         # Acoustic quality evaluation gates (SI-SDR, LSD, SC)
│   ├── state.py           # Clean CodecState memory containers
│   └── stateful.py        # True layer-by-layer stateful streaming engines
├── controller/            # Host session state machine and gating
│   ├── buffer.py          # Circular lookback buffer (Warmup & Burst replay)
│   ├── session.py         # SessionController orchestrating VAD & lazy decay
│   └── vad.py             # Lightweight two-stage CPU Voice Activity Detector
├── modules/               # Core neural network modules
│   ├── adapter.py         # Sub-250K Personalized Adapter & Full Pipeline
│   ├── bottleneck.py      # Bottleneck adapter layers
│   ├── causal_layers.py   # Causal dilated convolutions & residual blocks
│   ├── cleanser.py        # Speaker-invariant content cleanser with GRL
│   └── prosody.py         # In-graph prosody extraction head (F0 / Voicing)
├── runtime/               # Low-latency execution runtime
│   ├── cuda_graph.py      # CUDA Graph capturer and static memory runner
│   ├── ring_buffer.py     # Thread-safe lock-free audio circular ring buffer
│   └── wasapi.py          # WASAPI low-latency audio capture/playback wrapper
└── benchmark/             # Validation and profiling suites
    ├── contention.py      # Multi-tier concurrent contention benchmark
    ├── pareto.py          # Bottleneck compression and efficiency sweep
    └── report.py          # Automated scientific evaluation report generator
```

---

## 🛠️ Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/HotoKokoa1204/TimbreLite-Personalized-Ultra-Lightweight-Real-Time-Voice-Conversion.git
cd TimbreLite-Personalized-Ultra-Lightweight-Real-Time-Voice-Conversion

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Automated Test Suite

```bash
# Run all unit and integration tests
pytest tests -v

# Run with coverage report
pytest --cov=timbre_lite tests/
```

### Code Formatting and Linting

```bash
# Lint with Ruff
ruff check .

# Format with Ruff
ruff format .

# Strict type verification
mypy src tests
```

---

## 📊 Scientific Validation

TimbreLite's core architectural hypotheses are verified through rigorous automated test suites:

- **H1 (Parameter Budget)**: Adapter architecture achieves $\approx 107\text{K}$ parameters in 64-dim bottleneck configuration (verified $< 250\text{K}$ limit).
- **H2 (Prosody Decoupling)**: In-graph prosody head extracts $F_0$ and voicing within latent space (0 external CPU pitch tracker calls).
- **H3 (Contention Telemetry)**: Concurrent compute contention simulation exhibits $< 0.5\text{ ms}$ tail latency variance ($\Delta P99$) and $< 1.0\%$ 1% Low FPS degradation. Real-world DirectX/Vulkan game trace validation is conducted during hardware deployment.
- **H4 (Causal Streaming Invariant)**: Stateful causal codec achieves machine-precision streaming equivalence ($\max |y_{stream} - y_{batch}| < 1\times 10^{-4}$) and bit-exact counterfactual causality ($\Delta = 0.0$).
- **H5 (Zero-GPU Gating)**: Session controller issues `IDLE_SKIP` directives with 0 CUDA launches during silence and micro-pauses.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
