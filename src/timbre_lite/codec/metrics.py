"""Objective speech and audio quality evaluation metrics for codec gating."""

from __future__ import annotations

import torch


def calculate_si_sdr(ref: torch.Tensor, est: torch.Tensor) -> float:
    """Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels.

    Args:
        ref: Reference ground-truth audio tensor of shape (..., T).
        est: Estimated audio tensor of shape (..., T).

    Returns:
        SI-SDR score in dB. Higher is better.
    """
    assert ref.shape == est.shape, f"Shape mismatch: {ref.shape} vs {est.shape}"
    ref_zm = ref - torch.mean(ref, dim=-1, keepdim=True)
    est_zm = est - torch.mean(est, dim=-1, keepdim=True)

    dot = torch.sum(est_zm * ref_zm, dim=-1, keepdim=True)
    ref_energy = torch.sum(ref_zm**2, dim=-1, keepdim=True) + 1e-8
    scaling = dot / ref_energy

    target = scaling * ref_zm
    noise = est_zm - target

    target_pow = torch.sum(target**2, dim=-1) + 1e-8
    noise_pow = torch.sum(noise**2, dim=-1) + 1e-8

    sdr = 10.0 * torch.log10(target_pow / noise_pow)
    return float(torch.mean(sdr).item())


def calculate_spectral_convergence(
    ref: torch.Tensor,
    est: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
) -> float:
    """Calculate normalized Spectral Convergence between two audio signals.

    Args:
        ref: Reference audio tensor of shape (..., T).
        est: Estimated audio tensor of shape (..., T).
        n_fft: FFT window size.
        hop_length: Hop length between analysis frames.

    Returns:
        Spectral convergence score. Lower is better (0 is identical).
    """
    window = torch.hann_window(n_fft, device=ref.device)
    ref_stft = torch.stft(
        ref.view(-1, ref.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()
    est_stft = torch.stft(
        est.view(-1, est.shape[-1]),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    ).abs()

    diff_norm = torch.norm(ref_stft - est_stft, p="fro")
    ref_norm = torch.norm(ref_stft, p="fro") + 1e-8
    return float((diff_norm / ref_norm).item())


def calculate_log_spectral_distance(
    ref: torch.Tensor,
    est: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 128,
) -> float:
    """Calculate Log-Spectral Distance (LSD) in decibels.

    Args:
        ref: Reference audio tensor of shape (..., T).
        est: Estimated audio tensor of shape (..., T).
        n_fft: FFT window size.
        hop_length: Hop length between analysis frames.

    Returns:
        Log-Spectral Distance in dB. Lower is better.
    """
    window = torch.hann_window(n_fft, device=ref.device)
    ref_stft = (
        torch.stft(
            ref.view(-1, ref.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()
        ** 2
    )
    est_stft = (
        torch.stft(
            est.view(-1, est.shape[-1]),
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()
        ** 2
    )

    log_ratio = 10.0 * torch.log10((ref_stft + 1e-8) / (est_stft + 1e-8))
    lsd_per_frame = torch.sqrt(torch.mean(log_ratio**2, dim=-2))
    return float(torch.mean(lsd_per_frame).item())


def evaluate_codec_quality_gate(
    ref: torch.Tensor,
    y_quantized: torch.Tensor,
    y_continuous: torch.Tensor,
    max_relative_lsd_deg_db: float = 1.0,
    max_relative_sc_deg: float = 0.05,
) -> dict[str, float | bool]:
    """Evaluate multi-metric relative degradation for continuous bypass.

    Args:
        ref: Original ground truth input audio tensor.
        y_quantized: Output reconstructed via the standard vector quantizer.
        y_continuous: Output reconstructed by bypassing quantizer into decoder.
        max_relative_lsd_deg_db: Max allowed degradation in LSD (dB).
        max_relative_sc_deg: Max allowed degradation in Spectral Convergence.

    Returns:
        Dictionary containing measured metrics and gate pass/fail boolean.
    """
    sdr_quant = calculate_si_sdr(ref, y_quantized)
    sdr_cont = calculate_si_sdr(ref, y_continuous)

    sc_quant = calculate_spectral_convergence(ref, y_quantized)
    sc_cont = calculate_spectral_convergence(ref, y_continuous)

    lsd_quant = calculate_log_spectral_distance(ref, y_quantized)
    lsd_cont = calculate_log_spectral_distance(ref, y_continuous)

    rel_lsd_diff = calculate_log_spectral_distance(y_quantized, y_continuous)

    delta_sdr = sdr_cont - sdr_quant
    delta_sc = sc_cont - sc_quant
    delta_lsd = lsd_cont - lsd_quant

    # Gate passes if continuous bypass does not degrade relative to baseline
    passes_gate = (delta_lsd <= max_relative_lsd_deg_db) and (
        delta_sc <= max_relative_sc_deg
    )

    return {
        "si_sdr_quantized_db": sdr_quant,
        "si_sdr_continuous_db": sdr_cont,
        "delta_si_sdr_db": delta_sdr,
        "spectral_convergence_quantized": sc_quant,
        "spectral_convergence_continuous": sc_cont,
        "delta_spectral_convergence": delta_sc,
        "log_spectral_distance_quantized_db": lsd_quant,
        "log_spectral_distance_continuous_db": lsd_cont,
        "delta_log_spectral_distance_db": delta_lsd,
        "relative_quant_to_cont_lsd_db": rel_lsd_diff,
        "gate_passed": bool(passes_gate),
    }
