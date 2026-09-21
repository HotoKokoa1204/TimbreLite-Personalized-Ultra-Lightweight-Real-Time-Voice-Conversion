"""Unit and regression tests for causal F0 tracking and Gate 0 verification."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf

from timbre_lite.modules.f0 import (
    CausalF0Tracker,
    F0TrackerConfig,
    GateVerificationError,
    compare_with_reference_tracker,
    compute_log_quantiles,
    compute_octave_jump_metric,
    compute_oracle_fold_ratios,
    export_production_f0_config,
)


def generate_synthetic_harmonic_tone(
    f0: float,
    duration_sec: float = 1.0,
    sr: int = 24000,
    num_harmonics: int = 3,
) -> np.ndarray:
    """Generate synthetic harmonic tone with known F0."""
    t = np.arange(int(duration_sec * sr)) / sr
    sig = np.zeros_like(t)
    for h in range(1, num_harmonics + 1):
        amp = 1.0 / h
        sig += amp * np.sin(2 * np.pi * h * f0 * t)
    sig = 0.5 * sig / np.max(np.abs(sig))
    return sig.astype(np.float32)


def test_causal_f0_tracker_streaming_parity() -> None:
    """G0-6: Verify chunk-by-chunk streaming matches batch bit-for-bit."""
    cfg = F0TrackerConfig(
        sample_rate=24000,
        hop_length=320,
        window_length=1280,
        f_min=65.0,
        f_max=380.0,
    )
    tracker_batch = CausalF0Tracker(cfg)
    tracker_chunk = CausalF0Tracker(cfg)

    audio = generate_synthetic_harmonic_tone(f0=120.0, duration_sec=2.0, sr=24000)

    f0_b, log_f0_b, delta_b, vuv_b = tracker_batch.process_utterance(audio)

    num_frames = (len(audio) + cfg.hop_length - 1) // cfg.hop_length
    f0_c = np.zeros(num_frames, dtype=np.float32)
    log_f0_c = np.zeros(num_frames, dtype=np.float32)
    delta_c = np.zeros(num_frames, dtype=np.float32)
    vuv_c = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start = i * cfg.hop_length
        chunk = audio[start : start + cfg.hop_length]
        if len(chunk) < cfg.hop_length:
            chunk = np.pad(chunk, (0, cfg.hop_length - len(chunk)))
        f0_i, log_f0_i, delta_i, vuv_i = tracker_chunk.process_chunk(chunk)
        f0_c[i] = f0_i
        log_f0_c[i] = log_f0_i
        delta_c[i] = delta_i
        vuv_c[i] = vuv_i

    assert np.all(f0_b == f0_c), "F0 trajectory differs"
    assert np.all(log_f0_b == log_f0_c), "log-F0 trajectory differs"
    assert np.all(delta_b == delta_c), "Delta log-F0 differs"
    assert np.all(vuv_b == vuv_c), "VUV mask differs"


def test_causal_f0_tracker_frequency_bounds() -> None:
    """Verify tracker respects frequency search bounds [f_min, f_max]."""
    cfg = F0TrackerConfig(
        sample_rate=24000,
        hop_length=320,
        window_length=1280,
        f_min=65.0,
        f_max=380.0,
    )
    tracker = CausalF0Tracker(cfg)

    low_audio = generate_synthetic_harmonic_tone(f0=40.0, duration_sec=1.0)
    f0_low, _, _, vuv_low = tracker.process_utterance(low_audio)
    voiced_low = f0_low[vuv_low > 0.5]
    if len(voiced_low) > 0:
        assert np.all((voiced_low >= cfg.f_min) & (voiced_low <= cfg.f_max))

    high_audio = generate_synthetic_harmonic_tone(f0=450.0, duration_sec=1.0)
    f0_high, _, _, vuv_high = tracker.process_utterance(high_audio)
    voiced_high = f0_high[vuv_high > 0.5]
    if len(voiced_high) > 0:
        assert np.all((voiced_high >= cfg.f_min) & (voiced_high <= cfg.f_max))


def test_causal_f0_tracker_known_frequencies() -> None:
    """G0-7: Verify accuracy within +-3 Hz on synthetic harmonic tones."""
    cfg = F0TrackerConfig(
        sample_rate=24000,
        hop_length=320,
        window_length=1280,
        f_min=65.0,
        f_max=380.0,
    )
    tracker = CausalF0Tracker(cfg)

    for target_f0 in [80.0, 120.0, 200.0, 320.0]:
        audio = generate_synthetic_harmonic_tone(
            f0=target_f0, duration_sec=1.5, sr=24000
        )
        f0_traj, _, _, vuv = tracker.process_utterance(audio)
        voiced = f0_traj[vuv > 0.5]
        assert len(voiced) > 0, f"No voiced frames detected for {target_f0} Hz"
        stable_voiced = voiced[4:] if len(voiced) > 4 else voiced
        measured_med = float(np.median(stable_voiced))
        assert abs(measured_med - target_f0) <= 3.0, (
            f"Expected {target_f0} +-3 Hz, got {measured_med:.2f} Hz"
        )


def test_octave_jump_metric_calculation() -> None:
    """G0-1: Verify adjacent frame octave jump rate calculation."""
    f0 = np.array([100.0, 100.0, 200.0, 200.0, 100.0, 100.0], dtype=np.float32)
    vuv = np.ones(6, dtype=np.float32)

    res = compute_octave_jump_metric(f0, vuv)
    assert res["total_transitions"] == 5
    assert res["halving_count"] == 1
    assert res["doubling_count"] == 1
    assert pytest.approx(res["octave_jump_rate"], 1e-4) == 2.0 / 5.0


def test_oracle_comparator_and_fold_ratios() -> None:
    """G0-2 & G0-3: Verify oracle GPE, MAE cents, and fold ratio calculations."""
    tracker_f0 = np.array([100.0, 100.0, 200.0, 50.0], dtype=np.float32)
    oracle_f0 = np.array([100.0, 105.0, 100.0, 100.0], dtype=np.float32)
    tracker_vuv = np.ones(4, dtype=np.float32)
    oracle_vuv = np.ones(4, dtype=np.float32)

    metrics = compare_with_reference_tracker(
        tracker_f0, tracker_vuv, oracle_f0, oracle_vuv
    )

    assert metrics["total_consensus_frames"] == 4
    assert metrics["gpe_frames"] == 2
    assert pytest.approx(metrics["gpe_rate"], 1e-4) == 0.50
    assert metrics["oracle_octave_disagreement_count"] == 2
    assert pytest.approx(metrics["oracle_octave_disagreement_rate"], 1e-4) == 0.50

    fold_ratios = compute_oracle_fold_ratios(
        tracker_f0, tracker_vuv, oracle_f0, oracle_vuv
    )
    assert fold_ratios["ratio_sub_0_75_count"] == 1
    assert fold_ratios["ratio_0_75_to_1_25_count"] == 2
    assert fold_ratios["ratio_super_1_25_count"] == 1


def test_log_domain_iqr_quantiles() -> None:
    """Hard Requirement 4: Ensure log_iqr is calculated from log-domain quantiles."""
    log_vals = np.array([4.0, 4.2, 4.4, 4.6, 4.8, 5.0, 5.2], dtype=np.float64)
    q25, q75 = np.percentile(log_vals, [25, 75])
    expected_iqr = q75 - q25

    res = compute_log_quantiles(log_vals)
    assert pytest.approx(res["log_iqr"], 1e-6) == expected_iqr
    assert pytest.approx(res["log_median"], 1e-6) == np.median(log_vals)


def test_production_config_exporter_enforces_gates(tmp_path: Path) -> None:
    """G0-8: Exporter succeeds only if Gate 0 criteria pass."""
    out_file = tmp_path / "f0_production_config.json"

    with pytest.raises(GateVerificationError, match="G0-1"):
        export_production_f0_config(
            source_med_hz=95.6,
            source_iqr_hz=35.0,
            source_med_log=4.56,
            source_iqr_log=0.36,
            target_med_hz=329.6,
            target_iqr_hz=151.4,
            target_med_log=5.80,
            target_iqr_log=0.46,
            e_oct_source=0.035,
            e_oct_target=0.005,
            benchmark_gpe=0.04,
            output_path=out_file,
        )

    cfg_data = export_production_f0_config(
        source_med_hz=95.6,
        source_iqr_hz=35.0,
        source_med_log=4.56,
        source_iqr_log=0.36,
        target_med_hz=329.6,
        target_iqr_hz=151.4,
        target_med_log=5.80,
        target_iqr_log=0.46,
        e_oct_source=0.006,
        e_oct_target=0.003,
        benchmark_gpe=0.045,
        output_path=out_file,
    )

    assert out_file.exists()
    assert cfg_data["gate_verified"] is True
    assert cfg_data["source"]["median_hz"] == 95.6
    assert cfg_data["target"]["median_hz"] == 329.6
    assert pytest.approx(cfg_data["scale_log_iqr"], 1e-4) == 0.46 / 0.36


def test_benchmark_regression_fixtures() -> None:
    """Hard Requirement 3: Benchmark utterances must track within bounds."""
    fixture_path = Path("tests/fixtures/f0_regression_fixtures.json")
    assert fixture_path.exists(), "F0 regression fixture file missing"
    fixtures = json.loads(fixture_path.read_text("utf-8"))

    cfg = F0TrackerConfig(
        sample_rate=24000,
        hop_length=320,
        window_length=1280,
        f_min=65.0,
        f_max=380.0,
    )
    tracker = CausalF0Tracker(cfg)

    total_consensus_all = 0
    total_gpe_all = 0

    for utt_id, spec in fixtures.items():
        wav_path = Path(f"data/my_voice/processed/24k/{utt_id}.wav")
        assert wav_path.exists(), f"Benchmark file {wav_path} not found"
        wav, sr = sf.read(str(wav_path))

        f0_causal, _, _, vuv_causal = tracker.process_utterance(wav)
        voiced_c = f0_causal[vuv_causal > 0.5]
        assert len(voiced_c) > 0

        measured_med = float(np.median(voiced_c))
        f_min = spec["expected_f0_min_hz"]
        f_max = spec["expected_f0_max_hz"]
        assert f_min <= measured_med <= f_max, (
            f"{utt_id}: measured {measured_med:.2f} Hz outside [{f_min}, {f_max}]"
        )
        assert abs(measured_med - spec["disallowed_harmonic_hz"]) > 20.0, (
            f"{utt_id}: regressed to harmonic {spec['disallowed_harmonic_hz']} Hz"
        )

        f0_oracle, _, _ = librosa.pyin(
            wav, fmin=65.0, fmax=380.0, sr=sr, hop_length=320
        )
        oracle_vuv = (~np.isnan(f0_oracle)).astype(np.float32)
        oracle_f0 = np.nan_to_num(f0_oracle, nan=0.0).astype(np.float32)

        comp = compare_with_reference_tracker(
            f0_causal, vuv_causal, oracle_f0, oracle_vuv
        )
        total_consensus_all += int(comp["total_consensus_frames"])
        total_gpe_all += int(comp["gpe_frames"])

    overall_benchmark_gpe = total_gpe_all / max(1, total_consensus_all)
    assert overall_benchmark_gpe < 0.10, (
        f"G0-3 Failed: benchmark GPE {overall_benchmark_gpe * 100:.2f}% >= 10.0%"
    )
