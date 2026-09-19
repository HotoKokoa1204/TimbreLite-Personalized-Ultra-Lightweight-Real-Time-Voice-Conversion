"""Pareto frontier compression suite evaluating adapter bottleneck dimensions."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from timbre_lite.modules.adapter import PersonalizedAdapter


@dataclass
class ParetoDataPoint:
    """Benchmark measurements for a single adapter bottleneck configuration.

    Attributes:
        bottleneck_dim: Evaluated bottleneck dimension.
        parameter_count: Total trainable parameter count.
        mean_latency_ms: Average model execution latency in milliseconds.
        p95_latency_ms: 95th percentile execution latency.
        speaker_similarity: Speaker verification cosine similarity score.
        phonetic_preservation: Phonetic feature preservation score [0.0, 1.0].
    """

    bottleneck_dim: int
    parameter_count: int
    mean_latency_ms: float
    p95_latency_ms: float
    speaker_similarity: float | None = None
    phonetic_preservation: float | None = None


class ParetoCompressionSuite:
    """Evaluates adapter efficiency across diverse bottleneck dimensions.

    Sweeps through [128, 96, 64, 32, 16] dimensions to establish the empirical
    trade-off curve between parameter budget, latency, and representation quality.
    """

    DEFAULT_DIMENSIONS: list[int] = [128, 96, 64, 32, 16]

    def __init__(
        self,
        dimensions: list[int] | None = None,
        num_warmup: int = 5,
        num_iterations: int = 50,
    ) -> None:
        """Initialize Pareto compression suite.

        Args:
            dimensions: List of bottleneck dimensions to sweep.
            num_warmup: Number of warmup benchmark iterations.
            num_iterations: Number of timed execution iterations.
        """
        self.dimensions = (
            dimensions if dimensions is not None else self.DEFAULT_DIMENSIONS
        )
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations

    def evaluate_dimension(self, dim: int) -> ParetoDataPoint:
        """Evaluate a specific bottleneck dimension.

        Args:
            dim: Bottleneck channel dimension.

        Returns:
            ParetoDataPoint recording empirical measurements.
        """
        adapter = PersonalizedAdapter(
            in_dim=dim,
            out_dim=128,
            hidden_dim=dim,
            tcn_layers=4,
            gru_hidden=dim,
        )
        adapter.eval()
        param_count = adapter.count_parameters()

        dummy_chunk = torch.randn(1, dim, 1)
        state = adapter.init_state(batch_size=1)

        # Warmup
        for _ in range(self.num_warmup):
            _, state = adapter.forward_chunk(dummy_chunk, state)

        # Timing loop
        latencies: list[float] = []
        for _ in range(self.num_iterations):
            t0 = time.perf_counter()
            _, state = adapter.forward_chunk(dummy_chunk, state)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        sorted_l = sorted(latencies)
        mean_lat = sum(sorted_l) / len(sorted_l)
        p95_lat = sorted_l[int(len(sorted_l) * 0.95)]

        # Quality scores require trained persona model checkpoints evaluated on target
        # test sets; synthetic formulaic scores are prohibited for research integrity.
        return ParetoDataPoint(
            bottleneck_dim=dim,
            parameter_count=param_count,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            speaker_similarity=None,
            phonetic_preservation=None,
        )

    def run_sweep(self) -> list[ParetoDataPoint]:
        """Execute full Pareto compression sweep across all dimensions.

        Returns:
            List of ParetoDataPoint results ordered by descending dimension.
        """
        results: list[ParetoDataPoint] = []
        for dim in self.dimensions:
            dp = self.evaluate_dimension(dim)
            results.append(dp)
        return results
