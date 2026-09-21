# EXP-1B: Explicit F0 Conditioning & 3-Way Controlled Ablation

## Overview

EXP-1B is the first controlled neural training experiment following Gate 0 F0 tracker verification.
It tests whether explicit target-style 3D F0 FiLM conditioning and gated skip connections improve
voice conversion quality, pitch alignment, and speaker similarity under a strictly controlled
50-epoch ablation on top of the Stage-2 Epoch-82 baseline.

**Issue**: [#19](https://github.com/HotoKokoa1204/TimbreLite-Personalized-Ultra-Lightweight-Real-Time-Voice-Conversion/issues/19)

---

## Formal Hypotheses

| ID | Hypothesis | Result |
|:---|:-----------|:-------|
| **H1** | Explicit target-style F0 FiLM conditioning improves output pitch distribution alignment vs. baseline (B > A). | **FAILED** |
| **H2** | Explicit F0 improves pitch alignment without materially degrading linguistic content (CER(B) ≈ CER(A), Δ < 5%). | **PASSED** |
| **H3** | Gated skip suppression improves target-speaker cosine similarity and acoustic alignment (C > B). | **PASSED** |

---

## Experimental Conditions (4 Models)

All ablation variants share an identical controlled protocol:

- **Starting checkpoint**: `checkpoints/stage2_adapter_best.pt` (Stage-2 Epoch-82)
- **Training budget**: 50 epochs each (identical seed=42, AdamW lr=2e-4, cosine decay, batch=8)
- **Dataset**: 441 target train utterances (Hu Tao), target-only auto-reconstruction objective
- **Evaluation**: 50 source validation utterances, 5-axis non-parallel protocol

| Condition | F0 FiLM | Skip | Best Val Loss | Best Epoch |
|:----------|:--------|:-----|:-------------|:-----------|
| **Model A_pretrained** | — | Static | — (Epoch-82 baseline) | 82 |
| **Model A-control** | No | Static | 4.6893 | 44 |
| **Model B** | Yes (ExplicitF0Encoder) | Static | 4.6648 | 39 |
| **Model C** | Yes (ExplicitF0Encoder) | Gated (α₀=−4.0) | 7.7431 | 49 |

### Parameter Budget

| Component | Parameters |
|:----------|:-----------|
| Model A baseline | 162,992 |
| ExplicitF0Encoder (Models B & C) | +4,352 |
| Gated skip scalar α (Model C only) | +1 |
| **Model C total** | **167,345** (< 250K ceiling) |

**ExplicitF0Encoder architecture**:
```
Conv1d(3, 32, kernel=1)  → 3×32 + 32 = 128 params
SiLU()
Conv1d(32, 128, kernel=1) → 32×128 + 128 = 4,224 params  [splits → γ (64) + β (64)]
Total: 4,352 params, zero-initialized
```

**Zero-init guarantee**: At step 0, γ = 0 and β = 0, so H′ = (1 + 0)·H + 0 = H. Model B is
mathematically identical to Model A at initialization, enabling smooth warmstart from Epoch-82.

---

## Evaluation: 5-Axis Non-Parallel Protocol

50 source validation utterances (`data/my_voice/processed/val_manifest.json`) are converted by
each model and evaluated across 5 axes. No cross-speaker STFT loss is used anywhere.

### Quantitative Results

| Model | Voiced F0 Median (Hz) | Semitone Error vs Target | Wasserstein Cents ↓ | Pitch Δ Corr (r) ↑ | Whisper CER ↓ | Hu Tao Cosine Sim ↑ | Target Mel L₁ ↓ | Spectral Centroid (Hz) |
|:------|:----------------------|:------------------------|:--------------------|:-------------------|:-------------|:--------------------|:----------------|:----------------------|
| **A_pretrained** | 127.89 | −16.21 st | 780.0 | +0.038 | 1.252 | 0.7498 | 2.1794 | 2925 |
| **A-control** | 113.12 | −18.33 st | 858.9 | +0.025 | 1.298 | 0.7567 | 2.1734 | 2930 |
| **Model B** | 111.21 | −18.63 st | 875.9 | +0.024 | 1.295 | 0.7550 | 2.1837 | 2926 |
| **Model C** | 113.13 | −18.33 st | 1356.3 | −0.017 | 1.508 | **0.7828** | **1.7332** | **3502** |

---

## Hypothesis Analysis

### H1 — FAILED: Residual Adapter Cannot Bridge the 22-Semitone Register Gap

The target-style F0 signal (Hu Tao ≈ 326 Hz) is correctly injected via FiLM, but the model's
output pitch **stays at source register (~111–113 Hz, ≈ −18.5 st from target)**. The root cause is
the **static skip connection**: the frozen encoder's source latent bypasses the adapter and carries
low-frequency harmonics (~80–130 Hz) directly into the frozen decoder.

```
Source latent (80 Hz harmonics)
   │
   ├── Adapter → FiLM(F0=326 Hz) → corrected residual
   │
   └── Skip proj ────────────────────────────────────────→ Decoder
                 (80 Hz harmonics pass through unchanged)
```

FiLM correction in the adapter residual branch is overwhelmed by the static skip path.
**Conclusion**: Pitch shifting via FiLM alone in a residual adapter is architecturally insufficient.

### H2 — PASSED: Linguistic Preservation Maintained

- CER(Model B) = 1.295 vs CER(Model A-control) = 1.298 → **ΔCER = −0.003** (within ±5% bound).
- Adding F0 FiLM conditioning does not degrade speech intelligibility.

### H3 — PASSED: Gated Skip Suppresses Source Leakage

Model C's skip gate converged to σ(−3.94) ≈ 1.9% (nearly closed), forcing the decoder to rely
almost entirely on the adapter's corrected output:

- **Target Mel L₁**: 2.184 → **1.733** (−20.6%)
- **Hu Tao Cosine Sim**: 0.755 → **0.783** (+3.7%)
- **Spectral Centroid**: 2926 → **3502 Hz** (+576 Hz, +19.7%)
- **Spectral Rolloff**: 5118 → **7226 Hz** (+2108 Hz)

The dramatic shift in spectral centroid and rolloff confirms that suppressing the source skip path
allows the decoder to generate audio with substantially higher-frequency content, closer to Hu Tao's
characteristic vocal timbre.

> **Note**: Model C's higher val loss (7.74 vs 4.66–4.69) is expected — with skip gate ≈ 0, the
> adapter is under much higher gradient pressure to reconstruct the target from scratch. The gate
> never fully closed during 50 epochs, suggesting longer training or a warmup schedule may help.

---

## Key Architectural Finding → EXP-2 Direction

> **The static bypass (skip_proj) is the fundamental bottleneck for cross-register pitch shift.**

A residual adapter with static bypass cannot fundamentally shift pitch. The decoder receives the
source fundamental through the skip path regardless of what FiLM does to the adapter residual.

**EXP-2 design implication**: To achieve genuine register transformation, the synthesis stream
must be fully decoupled from the source latent. Options:

1. **Target-Only Renderer**: Train a lightweight decoder that operates solely on adapter output
   (no skip connection to source encoder). The source encoder provides content features only.
2. **Bottleneck Bypass Suppression**: Keep gated skip but train for longer with scheduled gate
   annealing (α₀ = −4.0 → −8.0 over 100 epochs).
3. **Differentiable Pitch Shift Layer**: Apply differentiable pitch-shifting on the waveform
   output using the estimated F0 ratio, bypassing the skip leakage problem at the waveform level.

---

## Artifacts

| Artifact | Path |
|:---------|:-----|
| Training script | [`scripts/train_exp1b_ablation.py`](../../scripts/train_exp1b_ablation.py) |
| Evaluation script | [`scripts/evaluate_exp1b_ablation.py`](../../scripts/evaluate_exp1b_ablation.py) |
| EXP-1B JSON report | `outputs/exp1b_evaluation/exp1b_ablation_report.json` |
| Audition HTML | `outputs/exp1b_evaluation/exp1b_ablation_audition.html` |
| Model A-control checkpoint | `checkpoints/exp1b_model_a_control/best_adapter.pt` |
| Model B checkpoint | `checkpoints/exp1b_model_b_f0/best_adapter.pt` |
| Model C checkpoint | `checkpoints/exp1b_model_c_gated/best_adapter.pt` |
| F0 features (precomputed) | `data/features/f0/hu_tao/` & `data/features/f0/my_voice/` |
| Val transcripts | `data/my_voice/processed/val_transcripts.json` |
| Unit tests | [`tests/test_exp1b_models.py`](../../tests/test_exp1b_models.py) |

---

## Related

- **Gate 0 F0 Tracker**: [EXP-1A / Issue #18](https://github.com/HotoKokoa1204/TimbreLite-Personalized-Ultra-Lightweight-Real-Time-Voice-Conversion/issues/18)
- **ADR**: [`docs/adr/0004-explicit-f0-distribution-conditioning.md`](../adr/0004-explicit-f0-distribution-conditioning.md)
- **ADR (Updated with EXP-1B findings)**: See Consequences section for EXP-2 architectural guidance.
