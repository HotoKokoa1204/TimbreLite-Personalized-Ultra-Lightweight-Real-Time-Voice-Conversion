"""CUDA Graph inference runner with single and ping-pong execution graphs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn


class GraphMode(str, Enum):
    """Execution mode for model inference."""

    EAGER = "eager"
    SINGLE_GRAPH = "single_graph"
    PING_PONG_GRAPH = "ping_pong_graph"


@dataclass
class ExecutionMetrics:
    """Execution timing and latency telemetry metrics.

    Attributes:
        latencies_ms: List of recorded execution latencies in milliseconds.
        p50_ms: 50th percentile latency.
        p95_ms: 95th percentile latency.
        p99_ms: 99th percentile latency.
        max_ms: Maximum latency spike observed.
    """

    latencies_ms: list[float] = field(default_factory=list)
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0

    def compute_summary(self) -> None:
        """Compute percentile statistics from collected latencies."""
        if not self.latencies_ms:
            return
        sorted_l = sorted(self.latencies_ms)
        n = len(sorted_l)
        self.p50_ms = sorted_l[int(n * 0.50)]
        self.p95_ms = sorted_l[min(n - 1, int(n * 0.95))]
        self.p99_ms = sorted_l[min(n - 1, int(n * 0.99))]
        self.max_ms = sorted_l[-1]


class CUDAGraphRunner:
    """CUDA Graph runner executing inference with zero CPU launch overhead.

    Supports alternating Ping-Pong execution graphs (`Graph_Even`/`Graph_Odd`)
    and Single persistent graph execution. Seamlessly falls back to eager mode
    when running on CPU environments.
    """

    def __init__(
        self,
        module: nn.Module,
        input_shape: tuple[int, ...] = (1, 1, 320),
        device: torch.device | None = None,
        mode: GraphMode = GraphMode.PING_PONG_GRAPH,
    ) -> None:
        """Initialize CUDAGraphRunner.

        Args:
            module: Target neural transformation model.
            input_shape: Static input tensor shape.
            device: Execution torch device.
            mode: Graph execution strategy.
        """
        self.module = module
        self.input_shape = input_shape
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.mode = mode
        self.metrics = ExecutionMetrics()

        self._is_cuda = self.device.type == "cuda"
        self._flip = 0  # Ping-pong alternator

        # Pre-allocated static buffers
        self.static_input_a = torch.zeros(input_shape, device=self.device)
        self.static_input_b = torch.zeros(input_shape, device=self.device)
        self.static_output_a: torch.Tensor | None = None
        self.static_output_b: torch.Tensor | None = None

        self._graph_a: torch.cuda.CUDAGraph | None = None
        self._graph_b: torch.cuda.CUDAGraph | None = None

        if self._is_cuda and mode != GraphMode.EAGER:
            self._capture_graphs()

    def _capture_graphs(self) -> None:
        """Capture CUDA graphs for static execution."""
        if not self._is_cuda:
            return

        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.module(self.static_input_a)
        torch.cuda.current_stream().wait_stream(s)

        # Capture Graph A
        self._graph_a = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph_a):
            self.static_output_a = self.module(self.static_input_a)

        if self.mode == GraphMode.PING_PONG_GRAPH:
            # Capture Graph B
            self._graph_b = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph_b):
                self.static_output_b = self.module(self.static_input_b)

    def execute_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Execute single inference step recording wall-clock runner latency.

        Note: Recorded timing measures end-to-end wall-clock runner latency
        (including tensor copy, graph replay, and synchronization). Pure
        asynchronous kernel execution time on GPU is profiled via CUDA Events.

        Args:
            chunk: Input audio chunk tensor matching input_shape.

        Returns:
            Model output tensor.
        """
        t0 = time.perf_counter()

        if self._is_cuda and self._graph_a is not None:
            if self.mode == GraphMode.PING_PONG_GRAPH and self._graph_b is not None:
                if self._flip == 0:
                    self.static_input_a.copy_(chunk)
                    self._graph_a.replay()
                    out = self.static_output_a
                    self._flip = 1
                else:
                    self.static_input_b.copy_(chunk)
                    self._graph_b.replay()
                    out = self.static_output_b
                    self._flip = 0
            else:
                self.static_input_a.copy_(chunk)
                self._graph_a.replay()
                out = self.static_output_a

            torch.cuda.synchronize()
            assert out is not None
            result = out.clone()
        else:
            # Eager mode execution
            with torch.no_grad():
                result = self.module(chunk)

        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        self.metrics.latencies_ms.append(elapsed_ms)
        return result

    def get_metrics_summary(self) -> ExecutionMetrics:
        """Compute and return latency metrics summary.

        Returns:
            Populated ExecutionMetrics.
        """
        self.metrics.compute_summary()
        return self.metrics
