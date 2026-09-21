"""Deterministic causal and offline F0 extraction with robust normalization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as signal
import soundfile as sf
import torch
from torchaudio import transforms as ta_transforms


class GateVerificationError(Exception):
    """Exception raised when Gate 0 acceptance criteria are not satisfied."""


@dataclass
class F0TrackerConfig:
    """Configuration parameters for causal F0 tracking and voicing decision.

    Attributes:
        sample_rate: Audio sample rate (default 24000 Hz).
        hop_length: Frame hop size in samples (default 320 = 13.33ms).
        window_length: Causal analysis window in samples (default 1280 = 53.33ms).
        f_min: Minimum fundamental frequency to search in Hz.
        f_max: Maximum fundamental frequency to search in Hz.
        vuv_on_threshold: Periodicity threshold to transition to Voiced state.
        vuv_off_threshold: Periodicity threshold to transition to Unvoiced state.
        energy_threshold_db: RMS energy cutoff in dBFS for forcing unvoiced.
        max_interp_gap_frames: Maximum unvoiced gap length allowed for causal hold.
        delta_clamp: Absolute clamp limit for delta log-F0.
        octave_jump_penalty: Candidate peak score penalty for octave jump.
        continuity_bonus: Candidate peak score bonus for smooth pitch transitions.
    """

    sample_rate: int = 24000
    hop_length: int = 320
    window_length: int = 1280
    f_min: float = 65.0
    f_max: float = 380.0
    vuv_on_threshold: float = 0.40
    vuv_off_threshold: float = 0.30
    energy_threshold_db: float = -50.0
    max_interp_gap_frames: int = 2
    delta_clamp: float = 0.50
    octave_jump_penalty: float = 0.35
    continuity_bonus: float = 0.05


@dataclass
class F0Statistics:
    """Summary statistics for voiced fundamental frequency distribution.

    Attributes:
        q5: 5th percentile of F0 in Hz.
        q25: 25th percentile (Q1) of F0 in Hz.
        median: 50th percentile (Q2) of F0 in Hz.
        q75: 75th percentile (Q3) of F0 in Hz.
        q95: 95th percentile of F0 in Hz.
        iqr: Interquartile range (q75 - q25) in Hz.
        log_median: Median of log-F0 values (from log-domain quantiles).
        log_iqr: Interquartile range of log-F0 values (log_q75 - log_q25).
        log_q5: 5th percentile of log-F0.
        log_q25: 25th percentile of log-F0.
        log_q75: 75th percentile of log-F0.
        log_q95: 95th percentile of log-F0.
        voiced_ratio: Ratio of voiced frames to total evaluated frames.
        num_voiced_frames: Total number of valid voiced frames.
        total_frames: Total number of processed frames.
        octave_error_candidate_rate: Fraction of frames with ~2x or ~0.5x jumps.
        mean_voiced_segment_sec: Average voiced segment duration in seconds.
        mean_unvoiced_gap_sec: Average unvoiced gap duration in seconds.
    """

    q5: float
    q25: float
    median: float
    q75: float
    q95: float
    iqr: float
    log_median: float
    log_iqr: float
    log_q5: float
    log_q25: float
    log_q75: float
    log_q95: float
    voiced_ratio: float
    num_voiced_frames: int
    total_frames: int
    octave_error_candidate_rate: float
    mean_voiced_segment_sec: float
    mean_unvoiced_gap_sec: float


class CausalF0Tracker:
    """Causal pitch and voicing extractor based on normalized autocorrelation.

    Maintains a 1280-sample history FIFO (53.33ms = 4 chunks @ 24kHz) to extract F0
    with strictly 0 algorithmic lookahead. Employs Boersma window autocorrelation
    normalization (r_w(tau) / r_win(tau)), causal lag-continuity prior scoring,
    hysteresis voicing decision, and exact batch/streaming parity.
    """

    def __init__(self, config: F0TrackerConfig | None = None) -> None:
        """Initialize CausalF0Tracker.

        Args:
            config: Optional F0TrackerConfig configuration container.
        """
        self.config = config or F0TrackerConfig()
        self.sr = self.config.sample_rate
        self.hop = self.config.hop_length
        self.win = self.config.window_length

        # Valid lag range corresponding to f_max .. f_min (lag = sr / f)
        self.min_lag = int(np.floor(self.sr / self.config.f_max))
        self.max_lag = int(np.ceil(self.sr / self.config.f_min))

        self.energy_cutoff = 10.0 ** (self.config.energy_threshold_db / 20.0)

        # Precompute static Hanning window and its autocorrelation
        self._hann = np.hanning(self.win).astype(np.float32)
        self._n_fft = 1 << (self.win * 2 - 1).bit_length()
        spec_win = np.fft.rfft(self._hann, n=self._n_fft)
        corr_win = np.fft.irfft(spec_win * np.conj(spec_win), n=self._n_fft)[: self.win]
        self._corr_win = np.maximum(corr_win.astype(np.float32), 1e-6)

        # Causal streaming state
        self.buffer = np.zeros(self.win, dtype=np.float32)
        self.is_voiced = False
        self.prev_lag: float = 0.0
        self.last_emitted_log_f0: float = 0.0
        self.unvoiced_gap_count: int = 0
        self.is_voiced_prev: bool = False

    def reset(self) -> None:
        """Reset internal streaming buffer and voicing state."""
        self.buffer.fill(0.0)
        self.is_voiced = False
        self.prev_lag = 0.0
        self.last_emitted_log_f0 = 0.0
        self.unvoiced_gap_count = 0
        self.is_voiced_prev = False

    def _parabolic_interpolation(
        self, corr: np.ndarray, peak_idx: int
    ) -> tuple[float, float]:
        """Perform 3-point parabolic interpolation around integer lag peak.

        Args:
            corr: Autocorrelation array.
            peak_idx: Integer index of peak in corr.

        Returns:
            Tuple of (refined_fractional_lag, interpolated_peak_value).
        """
        if 0 < peak_idx < len(corr) - 1:
            alpha = float(corr[peak_idx - 1])
            beta = float(corr[peak_idx])
            gamma = float(corr[peak_idx + 1])
            denom = alpha - 2.0 * beta + gamma
            if abs(denom) > 1e-12:
                delta = 0.5 * (alpha - gamma) / denom
                delta = max(-0.5, min(0.5, delta))
                val = beta - 0.25 * (alpha - gamma) * delta
                return float(peak_idx + delta), float(val)
        return float(peak_idx), float(corr[peak_idx])

    def _analyze_window(
        self, window: np.ndarray, prev_voiced: bool
    ) -> tuple[float, bool, float, float, float]:
        """Analyze a single 1280-sample causal window with window normalization.

        Args:
            window: 1D numpy array of length self.win.
            prev_voiced: Previous frame voiced boolean state for hysteresis.

        Returns:
            Tuple of (raw_f0, voiced_flag, periodicity, rms_energy, refined_lag).
        """
        rms = float(np.sqrt(np.mean(window**2) + 1e-12))
        if rms < self.energy_cutoff:
            return 0.0, False, 0.0, rms, 0.0

        # Mean centering and windowing
        w_centered = window - np.mean(window)
        w_tapered = w_centered * self._hann

        # Cross-correlation via FFT
        spec = np.fft.rfft(w_tapered, n=self._n_fft)
        corr = np.fft.irfft(spec * np.conj(spec), n=self._n_fft)[: self.win]

        r0 = float(corr[0])
        if r0 < 1e-9:
            return 0.0, False, 0.0, rms, 0.0

        # Boersma normalized autocorrelation: r_w(tau) / r_win(tau)
        norm_corr = (corr / self._corr_win) / (r0 / self._corr_win[0])

        search_slice = norm_corr[self.min_lag : self.max_lag + 1]
        if len(search_slice) == 0:
            return 0.0, False, 0.0, rms, 0.0

        # Find local peaks in search slice
        peak_indices, _ = signal.find_peaks(search_slice, height=0.20)
        if len(peak_indices) == 0:
            local_peak_idx = int(np.argmax(search_slice))
        else:
            # Score candidate peaks using correlation value + causal continuity prior
            best_score = -1e9
            local_peak_idx = peak_indices[0]
            for p in peak_indices:
                lag = self.min_lag + p
                val = float(search_slice[p])
                score = val
                if prev_voiced and self.prev_lag > 0.0:
                    ratio = lag / self.prev_lag
                    # Penalize candidates near octave jump intervals (0.5x, 2.0x)
                    if (0.45 <= ratio <= 0.55) or (1.80 <= ratio <= 2.20):
                        score -= self.config.octave_jump_penalty
                    elif abs(ratio - 1.0) < 0.15:
                        score += self.config.continuity_bonus
                if score > best_score:
                    best_score = score
                    local_peak_idx = p

        integer_lag = self.min_lag + local_peak_idx
        refined_lag, peak_val = self._parabolic_interpolation(norm_corr, integer_lag)
        periodicity = max(0.0, min(1.0, peak_val))

        # Hysteresis voicing decision
        if prev_voiced:
            voiced = periodicity >= self.config.vuv_off_threshold
        else:
            voiced = periodicity >= self.config.vuv_on_threshold

        f0 = (self.sr / refined_lag) if (voiced and refined_lag > 0) else 0.0
        # Clip to frequency bounds
        if f0 < self.config.f_min or f0 > self.config.f_max:
            voiced = False
            f0 = 0.0

        return float(f0), voiced, float(periodicity), rms, float(refined_lag)

    def process_chunk(
        self, chunk: np.ndarray | torch.Tensor
    ) -> tuple[float, float, float, float]:
        """Process a single 320-sample audio chunk strictly causally (lookahead = 0).

        Args:
            chunk: 1D audio chunk array or tensor of length self.hop (320 samples).

        Returns:
            Tuple of (f0, log_f0, delta_log_f0, vuv).
        """
        if isinstance(chunk, torch.Tensor):
            c = chunk.detach().cpu().numpy().astype(np.float32)
        else:
            c = np.asarray(chunk, dtype=np.float32)

        if c.ndim > 1:
            c = c.squeeze()

        if len(c) < self.hop:
            c = np.pad(c, (0, self.hop - len(c)))
        elif len(c) > self.hop:
            c = c[: self.hop]

        # Shift FIFO buffer by 320 samples
        self.buffer[: -self.hop] = self.buffer[self.hop :]
        self.buffer[-self.hop :] = c

        raw_f0, voiced, _, _, refined_lag = self._analyze_window(
            self.buffer, self.is_voiced
        )
        self.is_voiced = voiced

        clamp = self.config.delta_clamp

        if voiced:
            log_f0 = float(np.log(max(raw_f0, 1.0)))
            if self.is_voiced_prev:
                delta = max(-clamp, min(clamp, log_f0 - self.last_emitted_log_f0))
            else:
                delta = 0.0
            vuv = 1.0
            self.prev_lag = refined_lag
            self.last_emitted_log_f0 = log_f0
            self.unvoiced_gap_count = 0
            self.is_voiced_prev = True
            f0_out = raw_f0
        else:
            # Causal short unvoiced gap hold if gap <= max_interp_gap_frames
            if (
                self.unvoiced_gap_count < self.config.max_interp_gap_frames
                and self.is_voiced_prev
            ):
                log_f0 = self.last_emitted_log_f0
                f0_out = float(np.exp(log_f0))
                delta = 0.0
                vuv = 0.0
                self.unvoiced_gap_count += 1
            else:
                log_f0 = 0.0
                f0_out = 0.0
                delta = 0.0
                vuv = 0.0
                self.prev_lag = 0.0
                self.last_emitted_log_f0 = 0.0
                self.is_voiced_prev = False

        return f0_out, log_f0, delta, vuv

    def process_utterance(
        self, audio: np.ndarray | torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Process complete audio strictly simulating causal chunk streaming.

        Guarantees exact bitwise streaming parity with process_chunk.

        Args:
            audio: 1D audio waveform array or tensor at self.sr (24kHz).

        Returns:
            Tuple of (f0_trajectory, log_f0, delta_log_f0, vuv_mask).
        """
        if isinstance(audio, torch.Tensor):
            x = audio.detach().cpu().numpy().astype(np.float32)
        else:
            x = np.asarray(audio, dtype=np.float32)

        if x.ndim > 1:
            x = x.squeeze()

        self.reset()
        num_frames = (len(x) + self.hop - 1) // self.hop

        f0_traj = np.zeros(num_frames, dtype=np.float32)
        log_f0 = np.zeros(num_frames, dtype=np.float32)
        delta_log_f0 = np.zeros(num_frames, dtype=np.float32)
        vuv_mask = np.zeros(num_frames, dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop
            chunk = x[start : start + self.hop]
            f0_i, log_f0_i, delta_i, vuv_i = self.process_chunk(chunk)
            f0_traj[i] = f0_i
            log_f0[i] = log_f0_i
            delta_log_f0[i] = delta_i
            vuv_mask[i] = vuv_i

        return f0_traj, log_f0, delta_log_f0, vuv_mask


def compute_octave_jump_metric(
    f0_traj: np.ndarray, vuv: np.ndarray
) -> dict[str, float | int]:
    """Calculate adjacent-frame octave jump metrics on continuous voiced frames.

    Formula:
        r_t = F0_t / F0_{t-1} for vuv_t == 1 and vuv_{t-1} == 1
        halving candidate: r_t in [0.45, 0.55]
        doubling candidate: r_t in [1.80, 2.20]

    Args:
        f0_traj: 1D array of F0 in Hz.
        vuv: 1D binary voicing mask.

    Returns:
        Dict with total_transitions, halving_count, doubling_count, octave_jump_rate.
    """
    total_transitions = 0
    halving_count = 0
    doubling_count = 0

    for i in range(1, len(vuv)):
        if vuv[i] > 0.5 and vuv[i - 1] > 0.5 and f0_traj[i - 1] > 0:
            total_transitions += 1
            ratio = float(f0_traj[i] / f0_traj[i - 1])
            if 0.45 <= ratio <= 0.55:
                halving_count += 1
            elif 1.80 <= ratio <= 2.20:
                doubling_count += 1

    jump_rate = (
        float((halving_count + doubling_count) / total_transitions)
        if total_transitions > 0
        else 0.0
    )
    return {
        "total_transitions": total_transitions,
        "halving_count": halving_count,
        "doubling_count": doubling_count,
        "octave_jump_rate": jump_rate,
    }


def compare_with_reference_tracker(
    tracker_f0: np.ndarray,
    tracker_vuv: np.ndarray,
    oracle_f0: np.ndarray,
    oracle_vuv: np.ndarray,
) -> dict[str, float | int]:
    """Compare causal tracker F0 with high-precision offline oracle.

    Calculates Gross Pitch Error (GPE), MAE in cents, and oracle octave disagreement.

    Args:
        tracker_f0: Tracker F0 trajectory in Hz.
        tracker_vuv: Tracker voicing indicator (1.0 = voiced).
        oracle_f0: Reference oracle F0 trajectory in Hz.
        oracle_vuv: Reference oracle voicing indicator.

    Returns:
        Dictionary of consensus frames, GPE rate, MAE cents, and disagreement rate.
    """
    length = min(len(tracker_f0), len(oracle_f0))
    t_f0 = tracker_f0[:length]
    t_vuv = tracker_vuv[:length]
    o_f0 = oracle_f0[:length]
    o_vuv = oracle_vuv[:length]

    # Consensus voiced frames where both tracker and oracle detect voicing
    consensus_mask = (t_vuv > 0.5) & (o_vuv > 0.5) & (o_f0 > 0.0) & (t_f0 > 0.0)
    total_consensus = int(np.sum(consensus_mask))

    if total_consensus == 0:
        return {
            "total_consensus_frames": 0,
            "gpe_frames": 0,
            "gpe_rate": 0.0,
            "mae_cents": 0.0,
            "oracle_octave_disagreement_count": 0,
            "oracle_octave_disagreement_rate": 0.0,
        }

    c_t = t_f0[consensus_mask]
    c_o = o_f0[consensus_mask]

    ratios = c_t / c_o
    deviations = np.abs(ratios - 1.0)
    gpe_mask = deviations > 0.20
    gpe_frames = int(np.sum(gpe_mask))
    gpe_rate = float(gpe_frames / total_consensus)

    # MAE in cents on consensus voiced frames within +-20% GPE boundary
    non_gpe_mask = ~gpe_mask
    if np.any(non_gpe_mask):
        cents_errors = np.abs(1200.0 * np.log2(c_t[non_gpe_mask] / c_o[non_gpe_mask]))
        mae_cents = float(np.mean(cents_errors))
    else:
        mae_cents = 0.0

    # Oracle octave disagreement (~0.5x halving or ~2.0x doubling against oracle)
    oracle_octave_mask = (ratios >= 0.45) & (ratios <= 0.55) | (ratios >= 1.80) & (
        ratios <= 2.20
    )
    oracle_octave_count = int(np.sum(oracle_octave_mask))
    oracle_octave_rate = float(oracle_octave_count / total_consensus)

    return {
        "total_consensus_frames": total_consensus,
        "gpe_frames": gpe_frames,
        "gpe_rate": gpe_rate,
        "mae_cents": mae_cents,
        "oracle_octave_disagreement_count": oracle_octave_count,
        "oracle_octave_disagreement_rate": oracle_octave_rate,
    }


def compute_oracle_fold_ratios(
    tracker_f0: np.ndarray,
    tracker_vuv: np.ndarray,
    oracle_f0: np.ndarray,
    oracle_vuv: np.ndarray,
) -> dict[str, float | int]:
    """Calculate Octave Fold Ratios comparing tracker F0 against oracle F0.

    Bins:
        ratio < 0.75
        0.75 <= ratio <= 1.25
        ratio > 1.25

    Args:
        tracker_f0: Tracker F0 trajectory in Hz.
        tracker_vuv: Tracker voicing indicator.
        oracle_f0: Oracle F0 trajectory in Hz.
        oracle_vuv: Oracle voicing indicator.

    Returns:
        Dict with counts and percentages for each fold ratio bin.
    """
    length = min(len(tracker_f0), len(oracle_f0))
    consensus_mask = (
        (tracker_vuv[:length] > 0.5)
        & (oracle_vuv[:length] > 0.5)
        & (oracle_f0[:length] > 0.0)
        & (tracker_f0[:length] > 0.0)
    )
    total = int(np.sum(consensus_mask))
    if total == 0:
        return {
            "total_frames": 0,
            "ratio_sub_0_75_count": 0,
            "ratio_sub_0_75_rate": 0.0,
            "ratio_0_75_to_1_25_count": 0,
            "ratio_0_75_to_1_25_rate": 0.0,
            "ratio_super_1_25_count": 0,
            "ratio_super_1_25_rate": 0.0,
        }

    ratios = tracker_f0[:length][consensus_mask] / oracle_f0[:length][consensus_mask]
    sub_075 = int(np.sum(ratios < 0.75))
    mid_075_125 = int(np.sum((ratios >= 0.75) & (ratios <= 1.25)))
    super_125 = int(np.sum(ratios > 1.25))

    return {
        "total_frames": total,
        "ratio_sub_0_75_count": sub_075,
        "ratio_sub_0_75_rate": float(sub_075 / total),
        "ratio_0_75_to_1_25_count": mid_075_125,
        "ratio_0_75_to_1_25_rate": float(mid_075_125 / total),
        "ratio_super_1_25_count": super_125,
        "ratio_super_1_25_rate": float(super_125 / total),
    }


def compute_log_quantiles(
    log_f0_values: np.ndarray | list[float],
) -> dict[str, float]:
    """Compute exact quantiles and IQR directly in the log-F0 domain.

    Enforces Hard Requirement 4:
        IQR_log = Q75(log F0) - Q25(log F0)
        (Never log(IQR_F0)).

    Args:
        log_f0_values: Sequence of natural log-F0 values.

    Returns:
        Dict with log_q5, log_q25, log_median, log_q75, log_q95, log_iqr.
    """
    arr = np.asarray(log_f0_values, dtype=np.float64)
    if len(arr) == 0:
        return {
            "log_q5": 0.0,
            "log_q25": 0.0,
            "log_median": 0.0,
            "log_q75": 0.0,
            "log_q95": 0.0,
            "log_iqr": 0.0,
        }
    q5, q25, med, q75, q95 = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "log_q5": float(q5),
        "log_q25": float(q25),
        "log_median": float(med),
        "log_q75": float(q75),
        "log_q95": float(q95),
        "log_iqr": float(q75 - q25),
    }


def compute_dataset_f0_statistics(
    tracker: CausalF0Tracker,
    audio_paths: Sequence[str | Path],
    sr: int = 24000,
) -> tuple[F0Statistics, list[float], list[float]]:
    """Compute comprehensive frame-pooled voiced F0 statistics across audio files.

    Args:
        tracker: Configured CausalF0Tracker instance.
        audio_paths: Sequence of paths to 24kHz audio files.
        sr: Expected audio sample rate.

    Returns:
        Tuple of (F0Statistics, all_voiced_f0_values, all_voiced_log_f0_values).
    """
    all_voiced_f0: list[float] = []
    all_voiced_log_f0: list[float] = []
    total_frames = 0
    total_adjacent_voiced_pairs = 0
    octave_jump_count = 0

    voiced_durations: list[float] = []
    unvoiced_durations: list[float] = []
    frame_dur = tracker.hop / sr

    for path in audio_paths:
        data, file_sr = sf.read(str(path))
        if file_sr != sr:
            t_data = torch.from_numpy(data).float().unsqueeze(0)
            resampler = ta_transforms.Resample(orig_freq=file_sr, new_freq=sr)
            data = resampler(t_data).squeeze(0).numpy()

        f0_traj, log_f0, _, vuv = tracker.process_utterance(data)
        total_frames += len(vuv)

        # Track contiguous voiced and unvoiced segments
        curr_state = int(vuv[0]) if len(vuv) > 0 else 0
        curr_len = 1
        for i in range(1, len(vuv)):
            state = int(vuv[i])
            if state == curr_state:
                curr_len += 1
            else:
                if curr_state == 1:
                    voiced_durations.append(curr_len * frame_dur)
                else:
                    unvoiced_durations.append(curr_len * frame_dur)
                curr_state = state
                curr_len = 1
        if curr_len > 0:
            if curr_state == 1:
                voiced_durations.append(curr_len * frame_dur)
            else:
                unvoiced_durations.append(curr_len * frame_dur)

        # Calculate octave jumps using standardized metric
        oct_res = compute_octave_jump_metric(f0_traj, vuv)
        total_adjacent_voiced_pairs += int(oct_res["total_transitions"])
        octave_jump_count += int(oct_res["halving_count"]) + int(
            oct_res["doubling_count"]
        )

        voiced_mask = vuv > 0.5
        v_f0 = f0_traj[voiced_mask].tolist()
        v_log = log_f0[voiced_mask].tolist()
        all_voiced_f0.extend(v_f0)
        all_voiced_log_f0.extend(v_log)

    if len(all_voiced_f0) == 0:
        return (
            F0Statistics(
                q5=0.0,
                q25=0.0,
                median=0.0,
                q75=0.0,
                q95=0.0,
                iqr=0.0,
                log_median=0.0,
                log_iqr=0.0,
                log_q5=0.0,
                log_q25=0.0,
                log_q75=0.0,
                log_q95=0.0,
                voiced_ratio=0.0,
                num_voiced_frames=0,
                total_frames=total_frames,
                octave_error_candidate_rate=0.0,
                mean_voiced_segment_sec=0.0,
                mean_unvoiced_gap_sec=0.0,
            ),
            [],
            [],
        )

    f0_arr = np.array(all_voiced_f0, dtype=np.float64)
    q5, q25, med, q75, q95 = np.percentile(f0_arr, [5, 25, 50, 75, 95])
    iqr = q75 - q25

    # Compute log quantiles strictly in log-domain
    log_q = compute_log_quantiles(all_voiced_log_f0)

    octave_error_rate = (
        (octave_jump_count / total_adjacent_voiced_pairs)
        if total_adjacent_voiced_pairs > 0
        else 0.0
    )
    mean_voiced_sec = (
        float(np.mean(voiced_durations)) if len(voiced_durations) > 0 else 0.0
    )
    mean_unvoiced_sec = (
        float(np.mean(unvoiced_durations)) if len(unvoiced_durations) > 0 else 0.0
    )

    stats = F0Statistics(
        q5=float(q5),
        q25=float(q25),
        median=float(med),
        q75=float(q75),
        q95=float(q95),
        iqr=float(iqr),
        log_median=log_q["log_median"],
        log_iqr=log_q["log_iqr"],
        log_q5=log_q["log_q5"],
        log_q25=log_q["log_q25"],
        log_q75=log_q["log_q75"],
        log_q95=log_q["log_q95"],
        voiced_ratio=float(len(f0_arr) / max(total_frames, 1)),
        num_voiced_frames=len(f0_arr),
        total_frames=total_frames,
        octave_error_candidate_rate=float(octave_error_rate),
        mean_voiced_segment_sec=mean_voiced_sec,
        mean_unvoiced_gap_sec=mean_unvoiced_sec,
    )
    return stats, all_voiced_f0, all_voiced_log_f0


def compute_utterance_balanced_statistics(
    tracker: CausalF0Tracker,
    audio_paths: Sequence[str | Path],
    sr: int = 24000,
) -> dict[str, float]:
    """Compute utterance-balanced macro distribution statistics across audio files.

    Gives equal weighting to every utterance regardless of audio duration.

    Args:
        tracker: Configured CausalF0Tracker.
        audio_paths: Sequence of audio file paths.
        sr: Expected sample rate.

    Returns:
        Dictionary of utterance-balanced population metrics.
    """
    utt_medians_hz: list[float] = []
    utt_iqrs_hz: list[float] = []
    utt_medians_log: list[float] = []
    utt_iqrs_log: list[float] = []

    for path in audio_paths:
        data, file_sr = sf.read(str(path))
        if file_sr != sr:
            t_data = torch.from_numpy(data).float().unsqueeze(0)
            resampler = ta_transforms.Resample(orig_freq=file_sr, new_freq=sr)
            data = resampler(t_data).squeeze(0).numpy()

        f0_traj, log_f0, _, vuv = tracker.process_utterance(data)
        v_mask = vuv > 0.5
        v_f0 = f0_traj[v_mask]
        v_log = log_f0[v_mask]

        if len(v_f0) > 0:
            utt_medians_hz.append(float(np.median(v_f0)))
            q25_hz, q75_hz = np.percentile(v_f0, [25, 75])
            utt_iqrs_hz.append(float(q75_hz - q25_hz))

            q_log = compute_log_quantiles(v_log)
            utt_medians_log.append(q_log["log_median"])
            utt_iqrs_log.append(q_log["log_iqr"])

    if len(utt_medians_hz) == 0:
        return {
            "median_hz": 0.0,
            "iqr_hz": 0.0,
            "median_log_f0": 0.0,
            "iqr_log_f0": 0.0,
            "utterance_count": 0,
        }

    med_hz_arr = np.array(utt_medians_hz, dtype=np.float64)
    med_log_arr = np.array(utt_medians_log, dtype=np.float64)

    return {
        "median_hz": float(np.median(med_hz_arr)),
        "iqr_hz": float(np.mean(utt_iqrs_hz)),
        "median_log_f0": float(np.median(med_log_arr)),
        "iqr_log_f0": float(np.mean(utt_iqrs_log)),
        "utterance_count": len(utt_medians_hz),
    }


def map_source_f0_to_target(
    log_f0_src: np.ndarray | torch.Tensor,
    vuv: np.ndarray | torch.Tensor,
    source_stats: F0Statistics,
    target_stats: F0Statistics,
    pitch_bias_semitones: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source log-F0 trajectory to target vocal register via robust normalization.

    Formula:
        log_f0_out = med_t + (IQR_t / IQR_s) * (log_f0_s - med_s) + bias * (ln(2) / 12)

    Args:
        log_f0_src: Input source log-F0 array.
        vuv: Voiced/unvoiced binary mask.
        source_stats: F0Statistics of source speaker.
        target_stats: F0Statistics of target persona.
        pitch_bias_semitones: User-configurable pitch shift offset Delta k.

    Returns:
        Tuple of (mapped_log_f0, mapped_f0_hz).
    """
    if isinstance(log_f0_src, torch.Tensor):
        l_src = log_f0_src.detach().cpu().numpy().copy()
    else:
        l_src = np.asarray(log_f0_src).copy()

    if isinstance(vuv, torch.Tensor):
        mask = vuv.detach().cpu().numpy() > 0.5
    else:
        mask = np.asarray(vuv) > 0.5

    scale = target_stats.log_iqr / max(source_stats.log_iqr, 1e-6)
    bias_log = pitch_bias_semitones * (np.log(2.0) / 12.0)

    l_mapped = np.zeros_like(l_src)
    l_mapped[mask] = (
        target_stats.log_median
        + scale * (l_src[mask] - source_stats.log_median)
        + bias_log
    )

    f0_mapped = np.where(mask, np.exp(l_mapped), 0.0)
    return l_mapped, f0_mapped


def export_production_f0_config(
    source_med_hz: float,
    source_iqr_hz: float,
    source_med_log: float,
    source_iqr_log: float,
    target_med_hz: float,
    target_iqr_hz: float,
    target_med_log: float,
    target_iqr_log: float,
    e_oct_source: float,
    e_oct_target: float,
    benchmark_gpe: float,
    output_path: str | Path = "outputs/f0_production_config.json",
    pitch_bias_semitones: float = 0.0,
) -> dict[str, Any]:
    """Validate Gate 0 criteria and export decoupled production F0 configuration.

    Hard Gate Criteria:
        G0-1: e_oct_source < 0.01 (1.0%)
        G0-1: e_oct_target < 0.01 (1.0%)
        G0-3: benchmark_gpe < 0.10 (10.0%)

    Args:
        source_med_hz: Source median pitch in Hz.
        source_iqr_hz: Source pitch IQR in Hz.
        source_med_log: Source log-F0 median.
        source_iqr_log: Source log-F0 IQR.
        target_med_hz: Target median pitch in Hz.
        target_iqr_hz: Target pitch IQR in Hz.
        target_med_log: Target log-F0 median.
        target_iqr_log: Target log-F0 IQR.
        e_oct_source: Adjacent frame octave jump rate on source corpus.
        e_oct_target: Adjacent frame octave jump rate on target corpus.
        benchmark_gpe: Benchmark Gross Pitch Error against oracle.
        output_path: Path to save the production config JSON.
        pitch_bias_semitones: User-configurable pitch shift offset Delta k.

    Raises:
        GateVerificationError: If any of G0-1, G0-3 fails.

    Returns:
        Exported configuration dictionary.
    """
    if e_oct_source >= 0.01:
        msg = f"Gate G0-1 Failed: Source octave jump {e_oct_source * 100:.2f}% >= 1.0%"
        raise GateVerificationError(msg)
    if e_oct_target >= 0.01:
        msg = f"Gate G0-1 Failed: Target octave jump {e_oct_target * 100:.2f}% >= 1.0%"
        raise GateVerificationError(msg)
    if benchmark_gpe >= 0.10:
        msg = f"Gate G0-3 Failed: Benchmark GPE {benchmark_gpe * 100:.2f}% >= 10.0%"
        raise GateVerificationError(msg)

    scale_log_iqr = float(target_iqr_log / max(source_iqr_log, 1e-6))

    config_data: dict[str, Any] = {
        "source": {
            "median_hz": float(source_med_hz),
            "iqr_hz": float(source_iqr_hz),
            "median_log_f0": float(source_med_log),
            "iqr_log_f0": float(source_iqr_log),
        },
        "target": {
            "median_hz": float(target_med_hz),
            "iqr_hz": float(target_iqr_hz),
            "median_log_f0": float(target_med_log),
            "iqr_log_f0": float(target_iqr_log),
        },
        "scale_log_iqr": scale_log_iqr,
        "pitch_bias_semitones": float(pitch_bias_semitones),
        "gate_verified": True,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
    return config_data
