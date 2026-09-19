"""Three-tier gaming contention benchmark measuring frametime and FPS impact."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class ContentionLevel(str, Enum):
    """Contention benchmark severity levels."""

    LEVEL1_SYNTHETIC = "level1_synthetic_render_loop"
    LEVEL2_GPU_STRESS = "level2_gpu_compute_stress"
    LEVEL3_GAMING_TRACE = "level3_gaming_frametime_trace"


@dataclass
class ContentionMetrics:
    """Telemetry metrics recording game frametime impact under voice conversion.

    Attributes:
        target_fps: Target game refresh rate (e.g. 144 or 240 FPS).
        baseline_mean_ms: Baseline game frame time without VC active.
        baseline_p99_ms: Baseline 99th percentile frame time.
        active_mean_ms: Frame time during active voice conversion.
        active_p99_ms: 99th percentile frame time during active conversion.
        delta_p95_ms: Difference in 95th percentile frametime (target < 0.2ms).
        delta_p99_ms: Difference in 99th percentile frametime (target < 0.5ms).
        one_percent_low_drop_pct: Percentage drop in 1% Low FPS (target < 1.0%).
    """

    target_fps: int
    baseline_mean_ms: float
    baseline_p99_ms: float
    active_mean_ms: float
    active_p99_ms: float
    delta_p95_ms: float
    delta_p99_ms: float
    one_percent_low_drop_pct: float


class GamingContentionBenchmark:
    """Three-tier gaming contention benchmark suite.

    Quantifies competitive gaming contention to prove hypothesis H3:
    personal voice conversion introduces negligible (<1%) frametime overhead
    and imperceptible tail latency interference under high-refresh workloads.
    """

    def __init__(
        self,
        target_fps: int = 144,
        num_frames: int = 200,
    ) -> None:
        """Initialize GamingContentionBenchmark.

        Args:
            target_fps: Target game display frame rate (144 or 240).
            num_frames: Total simulated game frames to evaluate per test.
        """
        self.target_fps = target_fps
        self.target_frame_time_ms = 1000.0 / target_fps
        self.num_frames = num_frames

    def _simulate_game_loop(
        self,
        vc_worker: nn.Module | None = None,
        stress_level: ContentionLevel = ContentionLevel.LEVEL1_SYNTHETIC,
    ) -> list[float]:
        """Simulate game render loop with optional concurrent VC workload.

        Args:
            vc_worker: Optional neural transformation module representing VC.
            stress_level: Benchmark contention severity level.

        Returns:
            List of recorded game frame intervals in milliseconds.
        """
        frame_times: list[float] = []
        dummy_vc_chunk = torch.randn(1, 1, 320)

        # Baseline game workload tensor
        compute_size = 64 if stress_level == ContentionLevel.LEVEL1_SYNTHETIC else 128
        game_tensor = torch.randn(compute_size, compute_size)

        for i in range(self.num_frames):
            t0 = time.perf_counter()

            # 1. Game frame render compute
            _ = torch.matmul(game_tensor, game_tensor)

            # 2. If VC active, trigger audio chunk execution every ~8th frame (13.33ms)
            if vc_worker is not None and (i % 2 == 0):
                with torch.no_grad():
                    _ = vc_worker(dummy_vc_chunk)

            t1 = time.perf_counter()
            frame_times.append((t1 - t0) * 1000.0)

        return frame_times

    def run_benchmark(
        self,
        vc_module: nn.Module,
        level: ContentionLevel = ContentionLevel.LEVEL1_SYNTHETIC,
    ) -> ContentionMetrics:
        """Execute comparative contention benchmark between baseline and active VC.

        Args:
            vc_module: Target voice conversion module.
            level: Contention severity level.

        Returns:
            ContentionMetrics quantifying frametime variance and 1% low impact.
        """
        # 1. Baseline game run (VC idle / off)
        baseline_times = self._simulate_game_loop(vc_worker=None, stress_level=level)
        sorted_base = sorted(baseline_times)
        base_mean = sum(sorted_base) / len(sorted_base)
        base_p95 = sorted_base[int(len(sorted_base) * 0.95)]
        base_p99 = sorted_base[int(len(sorted_base) * 0.99)]

        # 2. Active VC run
        active_times = self._simulate_game_loop(vc_worker=vc_module, stress_level=level)
        sorted_act = sorted(active_times)
        act_mean = sum(sorted_act) / len(sorted_act)
        act_p95 = sorted_act[int(len(sorted_act) * 0.95)]
        act_p99 = sorted_act[int(len(sorted_act) * 0.99)]

        delta_p95 = max(0.0, act_p95 - base_p95)
        delta_p99 = max(0.0, act_p99 - base_p99)

        # 1% Low FPS: inverse of 99th percentile frametime
        base_1pct_fps = 1000.0 / (base_p99 + 1e-6)
        act_1pct_fps = 1000.0 / (act_p99 + 1e-6)
        drop_pct = max(0.0, ((base_1pct_fps - act_1pct_fps) / base_1pct_fps) * 100.0)

        return ContentionMetrics(
            target_fps=self.target_fps,
            baseline_mean_ms=base_mean,
            baseline_p99_ms=base_p99,
            active_mean_ms=act_mean,
            active_p99_ms=act_p99,
            delta_p95_ms=delta_p95,
            delta_p99_ms=delta_p99,
            one_percent_low_drop_pct=drop_pct,
        )
