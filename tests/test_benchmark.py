"""Automated test suite for Sub-Issue #15: Scientific Validation Benchmark Suite."""

from __future__ import annotations

import torch
import torch.nn as nn

from timbre_lite.benchmark.contention import (
    ContentionLevel,
    GamingContentionBenchmark,
)
from timbre_lite.benchmark.pareto import ParetoCompressionSuite
from timbre_lite.benchmark.report import ScientificReportGenerator


class MockVCModule(nn.Module):
    """Lightweight 1D Conv simulating active personal voice conversion."""

    def __init__(self) -> None:
        """Initialize mock module."""
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out: torch.Tensor = self.conv(x)
        return out


def test_pareto_compression_sweep() -> None:
    """Test 1: Run Pareto frontier sweep across [128, 96, 64, 32, 16] dimensions."""
    suite = ParetoCompressionSuite(
        dimensions=[128, 96, 64, 32, 16],
        num_warmup=2,
        num_iterations=10,
    )
    results = suite.run_sweep()

    assert len(results) == 5

    # Parameter counts must monotonically decrease
    prev_params = float("inf")
    for r in results:
        assert r.parameter_count < prev_params
        if r.bottleneck_dim <= 96:
            assert r.parameter_count < 250_000, (
                f"Compressed dim {r.bottleneck_dim} exceeded 250K: {r.parameter_count}"
            )
        assert r.speaker_similarity is None
        assert r.phonetic_preservation is None
        prev_params = r.parameter_count

    # 16-dim must be under 25K
    r16 = next(r for r in results if r.bottleneck_dim == 16)
    assert r16.parameter_count < 25_000

    # 64-dim standard must be ~107K
    r64 = next(r for r in results if r.bottleneck_dim == 64)
    assert 90_000 <= r64.parameter_count <= 125_000


def test_gaming_contention_benchmark() -> None:
    """Test 2: Run gaming contention benchmark and record frametime delta."""
    benchmark = GamingContentionBenchmark(target_fps=144, num_frames=50)
    vc_module = MockVCModule()

    metrics = benchmark.run_benchmark(
        vc_module=vc_module, level=ContentionLevel.LEVEL1_SYNTHETIC
    )

    assert metrics.target_fps == 144
    assert metrics.baseline_mean_ms > 0.0
    assert metrics.active_mean_ms > 0.0
    assert metrics.delta_p95_ms >= 0.0
    assert metrics.delta_p99_ms >= 0.0
    assert 0.0 <= metrics.one_percent_low_drop_pct <= 100.0


def test_scientific_validation_report_generation() -> None:
    """Test 3: Verify ScientificReportGenerator evaluates H1-H5 and formats Markdown."""
    suite = ParetoCompressionSuite(dimensions=[128, 64, 32], num_iterations=5)
    pareto_pts = suite.run_sweep()

    benchmark = GamingContentionBenchmark(target_fps=144, num_frames=30)
    metrics = benchmark.run_benchmark(vc_module=MockVCModule())

    generator = ScientificReportGenerator(
        pareto_points=pareto_pts, contention_metrics=metrics
    )
    hypotheses = generator.evaluate_hypotheses()

    assert len(hypotheses) == 5
    ids = [h.hypothesis_id for h in hypotheses]
    assert ids == ["H1", "H2", "H3", "H4", "H5"]

    report_md = generator.generate_markdown_report()
    assert "# TimbreLite Scientific Validation & Contention Report" in report_md
    assert "Pareto Frontier Compression Suite" in report_md
    assert "Gaming Contention Telemetry" in report_md
