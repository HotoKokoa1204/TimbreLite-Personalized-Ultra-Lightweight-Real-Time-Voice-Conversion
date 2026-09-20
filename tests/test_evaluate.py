"""Unit tests for scientific evaluation suite metrics and latency profiling."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from timbre_lite.benchmark.evaluate import (
    EvaluationSuite,
    compute_cer,
    compute_cosine_similarity,
    compute_levenshtein_distance,
    compute_wer,
    profile_streaming_latency,
)
from timbre_lite.inference.runner import VoiceConverter


def test_edit_distance_and_error_rates() -> None:
    """Verify Levenshtein distance, CER, and WER computations."""
    # Distance
    assert compute_levenshtein_distance("kitten", "sitting") == 3
    assert compute_levenshtein_distance("hello", "hello") == 0

    # CER
    assert compute_cer("胡桃語音", "胡桃語音") == 0.0
    assert compute_cer("胡桃語音", "胡桃聲音") == 0.25  # 1 char diff in 4 chars

    # WER
    assert compute_wer("enemy on the left", "enemy on the left") == 0.0
    assert (
        compute_wer("enemy on the left", "enemy on the right") == 0.25
    )  # 1 word diff in 4 words


def test_cosine_similarity_edge_cases() -> None:
    """Verify cosine similarity for parallel, orthogonal, and opposing vectors."""
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([2.0, 4.0, 6.0])
    v3 = torch.tensor([-1.0, -2.0, -3.0])
    v_orth = torch.tensor([0.0, -3.0, 2.0])

    sim_same = compute_cosine_similarity(v1, v2)
    assert abs(sim_same - 1.0) < 1e-5

    sim_opp = compute_cosine_similarity(v1, v3)
    assert abs(sim_opp - (-1.0)) < 1e-5

    sim_orth = compute_cosine_similarity(v1, v_orth)
    assert abs(sim_orth) < 1e-5


def test_profile_streaming_latency_and_evaluation_suite() -> None:
    """Verify streaming latency profiling and JSON report generation."""
    converter = VoiceConverter(device="cpu")
    perf = profile_streaming_latency(converter, num_chunks=5, warmup_chunks=2)

    assert "p50_ms" in perf
    assert "p95_ms" in perf
    assert "p99_ms" in perf
    assert "rtf" in perf
    assert perf["p50_ms"] > 0.0
    assert perf["rtf"] > 0.0

    suite = EvaluationSuite(converter)
    with tempfile.TemporaryDirectory() as tmp_dir:
        report_path = Path(tmp_dir) / "test_report.json"
        report = suite.evaluate_benchmark(
            output_report_path=report_path, num_latency_chunks=5
        )

        assert report_path.is_file()
        assert "latency" in report
        assert "architecture" in report
        assert report["architecture"]["sample_rate_hz"] == 24000
