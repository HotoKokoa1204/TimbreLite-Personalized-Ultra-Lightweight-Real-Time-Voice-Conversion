"""Benchmarking and scientific validation suite for TimbreLite."""

from timbre_lite.benchmark.contention import (
    ContentionLevel,
    ContentionMetrics,
    GamingContentionBenchmark,
)
from timbre_lite.benchmark.pareto import (
    ParetoCompressionSuite,
    ParetoDataPoint,
)
from timbre_lite.benchmark.report import (
    HypothesisResult,
    ScientificReportGenerator,
)

__all__ = [
    "ParetoDataPoint",
    "ParetoCompressionSuite",
    "ContentionLevel",
    "ContentionMetrics",
    "GamingContentionBenchmark",
    "HypothesisResult",
    "ScientificReportGenerator",
]
