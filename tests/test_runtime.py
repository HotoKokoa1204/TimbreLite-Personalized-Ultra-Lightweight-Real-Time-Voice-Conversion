"""Automated test suite for Sub-Issue #14: Low-Period WASAPI & Graph Runner."""

from __future__ import annotations

import threading
import time

import torch
import torch.nn as nn

from timbre_lite.runtime.cuda_graph import CUDAGraphRunner, GraphMode
from timbre_lite.runtime.ring_buffer import SPSCRingBuffer
from timbre_lite.runtime.wasapi import EngineStatus, WASAPIConfig, WASAPIEngine


class DummyAudioModule(nn.Module):
    """Simple 1D Conv module simulating inference workload."""

    def __init__(self) -> None:
        """Initialize dummy module."""
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        out: torch.Tensor = self.conv(x)
        return out


def test_spsc_ring_buffer_fifo_ordering() -> None:
    """Test 1: Verify FIFO ordering and data integrity in SPSCRingBuffer."""
    buffer = SPSCRingBuffer(capacity_frames=4, chunk_size=320)
    assert buffer.available_read == 0
    assert buffer.available_write == 4

    c0 = torch.ones(1, 1, 320) * 1.0
    c1 = torch.ones(1, 1, 320) * 2.0
    c2 = torch.ones(1, 1, 320) * 3.0

    assert buffer.push(c0)
    assert buffer.push(c1)
    assert buffer.push(c2)
    assert buffer.available_read == 3

    scratch = torch.zeros(1, 1, 320)
    out0 = buffer.pop(out_chunk=scratch)
    assert out0 is not None and torch.equal(out0, c0)

    out1 = buffer.pop()
    assert out1 is not None and torch.equal(out1, c1)

    out2 = buffer.pop()
    assert out2 is not None and torch.equal(out2, c2)

    assert buffer.available_read == 0


def test_spsc_ring_buffer_overrun_and_underrun_telemetry() -> None:
    """Test 2: Verify overrun and underrun telemetry accounting."""
    buffer = SPSCRingBuffer(capacity_frames=2, chunk_size=320)
    dummy = torch.randn(1, 1, 320)

    # Pop on empty -> underrun
    assert buffer.pop() is None
    assert buffer.underrun_count == 1

    # Fill buffer to capacity
    assert buffer.push(dummy)
    assert buffer.push(dummy)
    assert buffer.available_write == 0

    # Push on full -> overrun
    assert not buffer.push(dummy)
    assert buffer.overrun_count == 1


def test_spsc_ring_buffer_multithreaded_safety() -> None:
    """Test 3: Thread safety test across producer and consumer worker threads."""
    buffer = SPSCRingBuffer(capacity_frames=8, chunk_size=16)
    total_items = 100
    received_items: list[float] = []
    stop_event = threading.Event()

    def producer() -> None:
        for i in range(total_items):
            chunk = torch.ones(1, 1, 16) * float(i)
            while not buffer.push(chunk):
                time.sleep(0.0001)

    def consumer() -> None:
        while len(received_items) < total_items:
            popped = buffer.pop()
            if popped is not None:
                received_items.append(popped[0, 0, 0].item())
            else:
                time.sleep(0.0001)
        stop_event.set()

    t_prod = threading.Thread(target=producer)
    t_cons = threading.Thread(target=consumer)

    t_prod.start()
    t_cons.start()

    t_prod.join(timeout=5.0)
    t_cons.join(timeout=5.0)

    assert len(received_items) == total_items
    assert received_items == [float(i) for i in range(total_items)]


def test_cuda_graph_runner_eager_and_ping_pong() -> None:
    """Test 4: Verify CUDAGraphRunner latency tracking and execution."""
    module = DummyAudioModule()
    runner = CUDAGraphRunner(
        module=module,
        input_shape=(1, 1, 320),
        mode=GraphMode.PING_PONG_GRAPH,
    )

    chunk = torch.randn(1, 1, 320)
    for _ in range(20):
        out = runner.execute_chunk(chunk)
        assert out.shape == (1, 1, 320)
        assert not torch.isnan(out).any()

    metrics = runner.get_metrics_summary()
    assert len(metrics.latencies_ms) == 20
    assert metrics.p50_ms > 0.0
    assert metrics.p95_ms >= metrics.p50_ms


def test_wasapi_engine_stress_streaming_zero_underrun() -> None:
    """Test 5: Continuous 200-chunk streaming stress test with 0 underruns."""
    config = WASAPIConfig(chunk_size=320, elasticity_frames=2)
    engine = WASAPIEngine(config=config)
    engine.start()
    assert engine.status == EngineStatus.RUNNING

    module = DummyAudioModule()
    runner = CUDAGraphRunner(module=module, input_shape=(1, 1, 320))

    num_chunks = 200  # 2.67 seconds of audio
    dummy_input = torch.randn(1, 1, 320)

    for _ in range(num_chunks):
        # 1. Microphone capture callback
        success = engine.on_microphone_chunk(dummy_input)
        assert success, "Capture overrun"

        # 2. Processing thread
        captured = engine.input_buffer.pop()
        assert captured is not None
        converted = runner.execute_chunk(captured)
        engine.output_buffer.push(converted)

        # 3. Audio render callback
        rendered = engine.on_render_chunk()
        assert rendered is not None, "Render underrun occurred"
        assert rendered.shape == (1, 1, 320)

    engine.stop()
    assert engine.status.value == EngineStatus.STOPPED.value
    assert engine.total_underruns == 0
    assert engine.total_overruns == 0
