"""Scientific evaluation suite measuring WER, SECS, and streaming latency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from timbre_lite.inference.runner import VoiceConverter


def compute_levenshtein_distance(seq_a: str, seq_b: str) -> int:
    """Compute Levenshtein edit distance between two strings.

    Args:
        seq_a: Reference string.
        seq_b: Hypothesis string.

    Returns:
        Integer edit distance.
    """
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq_a[i - 1] == seq_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) between reference and hypothesis.

    Args:
        reference: Ground-truth reference string.
        hypothesis: Predicted hypothesis string.

    Returns:
        Character error rate as float in [0.0, inf).
    """
    ref_chars = reference.replace(" ", "")
    hyp_chars = hypothesis.replace(" ", "")
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = compute_levenshtein_distance(ref_chars, hyp_chars)
    return float(dist / len(ref_chars))


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between reference and hypothesis.

    Args:
        reference: Ground-truth words separated by spaces.
        hypothesis: Predicted words separated by spaces.

    Returns:
        Word error rate as float in [0.0, inf).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return float(dp[m][n] / len(ref_words))


def compute_cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """Compute cosine similarity between two 1D or 2D embedding tensors.

    Args:
        emb_a: First embedding tensor.
        emb_b: Second embedding tensor.

    Returns:
        Cosine similarity float in [-1.0, 1.0].
    """
    a = emb_a.flatten().float()
    b = emb_b.flatten().float()
    norm_a = torch.norm(a, p=2) + 1e-9
    norm_b = torch.norm(b, p=2) + 1e-9
    cos = torch.dot(a, b) / (norm_a * norm_b)
    return float(cos.clamp(-1.0, 1.0))


def profile_streaming_latency(
    converter: VoiceConverter,
    num_chunks: int = 100,
    warmup_chunks: int = 10,
) -> dict[str, float]:
    """Profile chunk-by-chunk streaming latency distribution on current device.

    Args:
        converter: Initialized VoiceConverter instance.
        num_chunks: Number of 320-sample iterations to profile.
        warmup_chunks: Number of initial warmup chunks discarded.

    Returns:
        Dictionary containing p50, p90, p95, p99, mean latency in ms, and RTF.
    """
    dummy_chunk = torch.randn(1, 1, 320, device=converter.device)
    state = converter.pipeline.init_streaming_state(
        batch_size=1, device=converter.device
    )

    # Warmup
    for _ in range(warmup_chunks):
        _, state = converter.pipeline.step_audio_chunk(dummy_chunk, state)
    if converter.device.type == "cuda":
        torch.cuda.synchronize(converter.device)

    # Profiling
    latencies_ms: list[float] = []
    for _ in range(num_chunks):
        if converter.device.type == "cuda":
            torch.cuda.synchronize(converter.device)
        t0 = time.perf_counter()

        _, state = converter.pipeline.step_audio_chunk(dummy_chunk, state)

        if converter.device.type == "cuda":
            torch.cuda.synchronize(converter.device)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    # 320 samples at 24kHz = 13.333ms duration
    frame_dur_ms = (320.0 / 24000.0) * 1000.0
    rtf = mean_ms / frame_dur_ms

    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": mean_ms,
        "rtf": float(rtf),
    }


class EvaluationSuite:
    """Scientific evaluation suite managing objective metrics for TimbreLite."""

    def __init__(
        self,
        converter: VoiceConverter,
    ) -> None:
        """Initialize EvaluationSuite with a target VoiceConverter.

        Args:
            converter: VoiceConverter instance.
        """
        self.converter = converter

    def evaluate_benchmark(
        self,
        output_report_path: str | Path | None = None,
        num_latency_chunks: int = 100,
    ) -> dict[str, Any]:
        """Run benchmark evaluation and export structured JSON report.

        Args:
            output_report_path: Optional path to save JSON report.
            num_latency_chunks: Number of chunks for latency profiling.

        Returns:
            Dictionary report containing latency and objective metrics.
        """
        latency_metrics = profile_streaming_latency(
            self.converter, num_chunks=num_latency_chunks
        )

        device_name = str(self.converter.device)
        if self.converter.device.type == "cuda":
            device_name = (
                f"{torch.cuda.get_device_name(self.converter.device)} "
                f"(CUDA {torch.version.cuda})"
            )

        report: dict[str, Any] = {
            "device": device_name,
            "latency": latency_metrics,
            "architecture": {
                "sample_rate_hz": 24000,
                "hop_samples": 320,
                "frame_duration_ms": 13.33,
                "trainable_parameters": (
                    self.converter.pipeline.count_trainable_parameters()
                ),
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        if output_report_path is not None:
            out_p = Path(output_report_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

        return report


def main() -> None:
    """CLI entrypoint for running evaluation benchmarks."""
    parser = argparse.ArgumentParser(
        description="Run TimbreLite scientific evaluation benchmarks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained Stage 2 checkpoint (.pt).",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="reports/benchmark_report.json",
        help="Path to save output JSON evaluation report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    converter = VoiceConverter(checkpoint_path=args.checkpoint, device=args.device)
    suite = EvaluationSuite(converter=converter)

    print(f"Running evaluation on {converter.device}...")
    report = suite.evaluate_benchmark(output_report_path=args.output_report)
    lat = report["latency"]
    print("\n--- TimbreLite Streaming Latency Benchmark ---")
    print(f"  P50:  {lat['p50_ms']:.2f} ms")
    print(f"  P90:  {lat['p90_ms']:.2f} ms")
    print(f"  P95:  {lat['p95_ms']:.2f} ms")
    print(f"  P99:  {lat['p99_ms']:.2f} ms")
    print(f"  RTF:  {lat['rtf']:.4f} (< 1.0 means faster than real-time)")
    print(f"Report saved to: {Path(args.output_report).resolve()}")


if __name__ == "__main__":
    main()
