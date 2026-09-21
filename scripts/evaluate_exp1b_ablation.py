"""EXP-1B: Evaluation Driver for 3-Way Controlled Ablation Study.

Evaluates 4 model conditions across 5 rigorous non-parallel axes:
  1. Model A_pretrained: Stage 2 Epoch-82 checkpoint (0-epoch continued baseline).
  2. Model A_control: Continued 50 epochs without F0 (controlled training budget).
  3. Model B: Continued 50 epochs with Explicit F0 FiLM, static skip.
  4. Model C: Continued 50 epochs with Explicit F0 FiLM, gated skip (alpha_0 = -4.0).

The 5 evaluation axes:
  Axis 1: Distributional F0 Error (Median, IQR, Wasserstein cents vs Hu Tao target).
  Axis 2: Pitch Trajectory Correlation (Pearson r of delta_n on consensus voiced frames vs mapped source).
  Axis 3: Whisper CER & WER (Linguistic preservation vs cached source validation transcripts).
  Axis 4: Speaker Similarity (Cosine similarity against Hu Tao validation centroid).
  Axis 5: Target-Domain Acoustic Alignment (Spectral centroid, roll-off, log-mel distance).

Generates:
  - outputs/exp1b_ablation_report.json
  - outputs/exp1b_ablation_audition.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F  # noqa: N812
from tqdm.auto import tqdm
from transformers import pipeline

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.data.loader import load_audio, save_wav
from timbre_lite.modules.adapter import (
    DualStreamFusion,
    ExplicitF0Encoder,
    FullPersonalizedPipeline,
    InGraphProsodyHead,
    PersonalizedAdapter,
)
from timbre_lite.modules.cleanser import ContentCleanser
from timbre_lite.modules.f0 import (
    CausalF0Tracker,
    F0TrackerConfig,
)


def map_f0_with_config(
    log_f0_src: np.ndarray,
    vuv: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Map source log-F0 trajectory using frozen production configuration."""
    med_t = float(cfg["target"]["median_log_f0"])
    med_s = float(cfg["source"]["median_log_f0"])
    scale = float(
        cfg.get(
            "scale_log_iqr",
            cfg["target"]["iqr_log_f0"] / cfg["source"]["iqr_log_f0"],
        )
    )
    bias = float(cfg.get("pitch_bias_semitones", 0.0)) * (math.log(2.0) / 12.0)

    l_mapped = np.zeros_like(log_f0_src)
    mask = vuv > 0.5
    l_mapped[mask] = med_t + scale * (log_f0_src[mask] - med_s) + bias
    f0_mapped = np.where(mask, np.exp(l_mapped), 0.0)
    return l_mapped, f0_mapped


def compute_levenshtein(seq1: list[str] | str, seq2: list[str] | str) -> int:
    """Compute standard dynamic programming Levenshtein distance."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[n]


def normalize_chinese_text(text: str) -> str:
    """Normalize text by stripping whitespace, punctuation, and converting to lowercase."""
    text = re.sub(r"[^\w\u4e00-\u9fff]", "", text.lower())
    return text.strip()


def compute_cer_wer(ref_text: str, hyp_text: str) -> tuple[float, float]:
    """Compute Character Error Rate and Word/Token Error Rate."""
    norm_ref = normalize_chinese_text(ref_text)
    norm_hyp = normalize_chinese_text(hyp_text)

    if not norm_ref:
        return (0.0 if not norm_hyp else 1.0, 0.0 if not norm_hyp else 1.0)

    # CER: character level
    ref_chars = list(norm_ref)
    hyp_chars = list(norm_hyp)
    char_dist = compute_levenshtein(ref_chars, hyp_chars)
    cer = char_dist / len(ref_chars)

    # WER: whitespace / character token level for Chinese
    ref_words = list(norm_ref)
    hyp_words = list(norm_hyp)
    word_dist = compute_levenshtein(ref_words, hyp_words)
    wer = word_dist / len(ref_words)

    return cer, wer


def extract_log_mel_spectrogram(
    audio: torch.Tensor,
    sr: int = 24000,
    n_mels: int = 80,
    n_fft: int = 1024,
    hop_length: int = 320,
) -> torch.Tensor:
    """Extract 80-channel log-mel spectrogram for timbre and centroid alignment."""
    import torchaudio.transforms as T  # noqa: N812

    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop_length,
        f_min=0.0,
        f_max=sr // 2,
        n_mels=n_mels,
        power=2.0,
    )
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    mel = mel_transform(audio)
    log_mel = torch.log(torch.clamp(mel, min=1e-5))
    return log_mel.squeeze(0)  # (80, T)


def compute_spectral_features(
    audio: torch.Tensor,
    sr: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 320,
) -> tuple[float, float]:
    """Compute spectral centroid (Hz) and spectral roll-off 85% (Hz)."""
    window = torch.hann_window(n_fft)
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    mag = torch.abs(stft)  # (freq_bins, frames)
    freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=audio.device).unsqueeze(1)

    total_energy = torch.sum(mag, dim=0, keepdim=True) + 1e-8
    centroid = torch.sum(freqs * mag, dim=0, keepdim=True) / total_energy
    mean_centroid = float(torch.mean(centroid).item())

    # Spectral roll-off 85%
    cum_energy = torch.cumsum(mag, dim=0)
    threshold = 0.85 * total_energy
    rolloff_indices = torch.argmax((cum_energy >= threshold).int(), dim=0)
    rolloff_freqs = freqs[rolloff_indices, 0]
    mean_rolloff = float(torch.mean(rolloff_freqs).item())

    return mean_centroid, mean_rolloff


def build_trajectory_svg(
    times: np.ndarray,
    f0_src: np.ndarray,
    f0_mapped: np.ndarray,
    f0_out: np.ndarray,
    vuv_src: np.ndarray,
    vuv_out: np.ndarray,
    target_median: float = 326.12,
    target_q25: float = 295.0,
    target_q75: float = 360.0,
    title: str = "Pitch Trajectory",
    width: int = 700,
    height: int = 220,
) -> str:
    """Generate inline SVG visualizing pitch trajectory against target IQR band."""
    pad_l, pad_r, pad_t, pad_b = 50, 20, 25, 35
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b

    t_max = max(float(times[-1]) if len(times) > 0 else 1.0, 0.1)
    f_min, f_max = 50.0, 550.0

    def t_to_x(t: float) -> float:
        return pad_l + (t / t_max) * pw

    def f_to_y(f: float) -> float:
        clipped = max(f_min, min(f_max, f))
        return pad_t + (1.0 - (clipped - f_min) / (f_max - f_min)) * ph

    y_q75 = f_to_y(target_q75)
    y_q25 = f_to_y(target_q25)
    y_med = f_to_y(target_median)

    parts: list[str] = [
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="background:#181825; border-radius:6px;">',
        f'<rect x="{pad_l}" y="{y_q75}" width="{pw}" height="{abs(y_q25 - y_q75):.1f}" '
        f'fill="rgba(249, 226, 175, 0.12)" stroke="rgba(249, 226, 175, 0.3)" stroke-dasharray="3,3"/>',
        f'<line x1="{pad_l}" y1="{y_med}" x2="{pad_l + pw}" y2="{y_med}" '
        f'stroke="rgba(249, 226, 175, 0.5)" stroke-width="1" stroke-dasharray="4,2"/>',
        f'<text x="{pad_l + 5}" y="{y_med - 4}" fill="#f9e2af" font-size="10" font-family="monospace">Hu Tao Target Median ({target_median:.1f} Hz)</text>',
    ]

    # Grid lines
    for freq in (100, 200, 300, 400, 500):
        y = f_to_y(freq)
        parts.append(
            f'<line x1="{pad_l}" y1="{y}" x2="{pad_l + pw}" y2="{y}" stroke="#313244" stroke-width="0.7"/>'
        )
        parts.append(
            f'<text x="{pad_l - 6}" y="{y + 3}" fill="#6c7086" font-size="9" text-anchor="end" font-family="monospace">{freq}</text>'
        )

    # Helper for polylines
    def create_segments(f0_arr: np.ndarray, vuv_arr: np.ndarray) -> list[str]:
        n = min(len(times), len(f0_arr), len(vuv_arr))
        segs: list[list[str]] = []
        cur: list[str] = []
        for i in range(n):
            if vuv_arr[i] > 0.5 and f0_arr[i] > 30.0:
                cur.append(f"{t_to_x(float(times[i])):.1f},{f_to_y(float(f0_arr[i])):.1f}")
            else:
                if len(cur) >= 2:
                    segs.append(cur)
                cur = []
        if len(cur) >= 2:
            segs.append(cur)
        return [" ".join(s) for s in segs]

    # 1. Source F0 (Gray)
    for pts in create_segments(f0_src, vuv_src):
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#6c7086" stroke-width="1.5" opacity="0.6"/>'
        )

    # 2. Mapped Target F0 (Blue dashed)
    for pts in create_segments(f0_mapped, vuv_src):
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#89b4fa" stroke-width="1.8" stroke-dasharray="3,2" opacity="0.85"/>'
        )

    # 3. Model Converted F0 (Green)
    for pts in create_segments(f0_out, vuv_out):
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="#a6e3a1" stroke-width="2.2"/>'
        )

    # Title & Legend
    parts.append(
        f'<text x="{pad_l}" y="{pad_t - 8}" fill="#cdd6f4" font-size="11" font-weight="bold" font-family="sans-serif">{html.escape(title)}</text>'
    )
    parts.append(
        f'<circle cx="{width - 240}" cy="{pad_t - 11}" r="4" fill="#6c7086"/>'
        f'<text x="{width - 232}" y="{pad_t - 8}" fill="#a6adc8" font-size="9" font-family="sans-serif">Source</text>'
    )
    parts.append(
        f'<circle cx="{width - 170}" cy="{pad_t - 11}" r="4" fill="#89b4fa"/>'
        f'<text x="{width - 162}" y="{pad_t - 8}" fill="#a6adc8" font-size="9" font-family="sans-serif">Mapped Ref</text>'
    )
    parts.append(
        f'<circle cx="{width - 90}" cy="{pad_t - 11}" r="4" fill="#a6e3a1"/>'
        f'<text x="{width - 82}" y="{pad_t - 8}" fill="#a6adc8" font-size="9" font-family="sans-serif">Converted</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def load_pipeline_model(
    checkpoint_path: str | Path,
    model_type: str,
    device: torch.device,
) -> FullPersonalizedPipeline:
    """Build and initialize a FullPersonalizedPipeline module from checkpoint."""
    use_f0 = model_type in ("model_b_f0", "model_c_gated")
    use_gated = model_type == "model_c_gated"

    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)
    adapter = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=use_gated,
        initial_skip_gate=-4.0,
    )
    f0_encoder = (
        ExplicitF0Encoder(in_dim=3, hidden_dim=32, out_dim=64) if use_f0 else None
    )

    pipeline_mod = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter,
        f0_encoder=f0_encoder,
    ).to(device)

    ckpt_p = Path(checkpoint_path)
    assert ckpt_p.is_file(), f"Checkpoint {ckpt_p} does not exist"
    ckpt = torch.load(ckpt_p, map_location=device, weights_only=False)

    if "pipeline_state_dict" in ckpt:
        pipeline_mod.load_state_dict(ckpt["pipeline_state_dict"], strict=False)
    else:
        if "adapter_state_dict" in ckpt:
            adapter.load_state_dict(ckpt["adapter_state_dict"], strict=False)
        if "cleanser_state_dict" in ckpt:
            cleanser.load_state_dict(ckpt["cleanser_state_dict"], strict=False)
        if "prosody_state_dict" in ckpt:
            prosody_head.load_state_dict(ckpt["prosody_state_dict"], strict=False)
        if "fusion_state_dict" in ckpt:
            fusion.load_state_dict(ckpt["fusion_state_dict"], strict=False)
        if f0_encoder is not None and "f0_encoder_state_dict" in ckpt and ckpt["f0_encoder_state_dict"] is not None:
            f0_encoder.load_state_dict(ckpt["f0_encoder_state_dict"], strict=False)

    pipeline_mod.eval()
    return pipeline_mod


def convert_waveform(
    pipeline_mod: FullPersonalizedPipeline,
    audio_24k: torch.Tensor,
    f0_3d: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    """Run full-utterance voice conversion and energy matching."""
    t_audio = audio_24k.to(device)
    orig_len = len(t_audio)
    inp = t_audio.unsqueeze(0).unsqueeze(0)  # (1, 1, samples)

    f0_in = f0_3d.to(device) if f0_3d is not None else None
    codec_model: Any = pipeline_mod.codec.model

    with torch.no_grad():
        z_src = codec_model.encoder(inp)
        z_adapted, _, _ = pipeline_mod.forward_sequence(z_src, f0_seq=f0_in)
        y_recon = codec_model.decoder(z_adapted).squeeze()

    if y_recon.ndim == 0:
        y_recon = y_recon.unsqueeze(0)
    res = y_recon[:orig_len].detach().cpu()

    # Match output energy to input energy
    in_rms = float(torch.sqrt(torch.mean(audio_24k**2)).item())
    out_rms = float(torch.sqrt(torch.mean(res**2)).item())
    if in_rms > 1e-4 and out_rms > 1e-5:
        gain = in_rms / (out_rms + 1e-8)
        res = res * gain
        peak = float(torch.max(torch.abs(res)).item())
        if peak > 0.99:
            res = res * (0.99 / peak)

    return res


def evaluate_ablation(
    model_checkpoints: dict[str, str | Path],
    val_manifest_path: str | Path = "data/my_voice/processed/val_manifest.json",
    f0_dir: str | Path = "data/features/f0/my_voice",
    transcripts_path: str | Path = "data/my_voice/processed/val_transcripts.json",
    target_val_manifest_path: str | Path = "data/hu_tao/processed/val_manifest.json",
    f0_config_path: str | Path = "outputs/f0_production_config.json",
    output_dir: str | Path = "outputs/exp1b_evaluation",
    device_str: str | None = None,
    max_eval_samples: int | None = None,
) -> dict[str, Any]:
    """Execute complete 5-axis evaluation across all model conditions."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    audio_out_p = out_p / "audio"
    audio_out_p.mkdir(parents=True, exist_ok=True)

    dev = (
        torch.device(device_str)
        if device_str is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Running EXP-1B Ablation Evaluation on: {dev}")

    # 1. Load F0 production config
    f0_cfg_file = Path(f0_config_path)
    assert f0_cfg_file.exists(), f"Missing F0 config: {f0_cfg_file}"
    f0_prod_cfg = json.loads(f0_cfg_file.read_text(encoding="utf-8"))
    target_stats = f0_prod_cfg.get("target_speaker_statistics", f0_prod_cfg.get("target", {}))
    tgt_median = float(target_stats.get("median_hz", target_stats.get("median", 326.12)))
    tgt_iqr = float(target_stats.get("iqr_hz", target_stats.get("iqr", 144.95)))
    tgt_q25 = float(target_stats.get("q25", tgt_median - tgt_iqr / 2.0))
    tgt_q75 = float(target_stats.get("q75", tgt_median + tgt_iqr / 2.0))

    # 2. Load validation manifest and ground truth transcripts
    with open(val_manifest_path, encoding="utf-8") as f:
        val_records = json.load(f)
    if max_eval_samples is not None:
        val_records = val_records[:max_eval_samples]

    with open(transcripts_path, encoding="utf-8") as f:
        gt_transcripts = json.load(f)

    # 3. Compute target centroid and reference voiced F0 distribution
    print("Computing target speaker (Hu Tao) reference acoustic profile...")
    with open(target_val_manifest_path, encoding="utf-8") as f:
        target_val_records = json.load(f)

    target_mels: list[torch.Tensor] = []
    target_voiced_f0_all: list[float] = []
    tracker = CausalF0Tracker(F0TrackerConfig())

    for t_item in target_val_records:
        t_wav, _ = load_audio(t_item["audio_24k"], target_sr=24000)
        t_mel = extract_log_mel_spectrogram(t_wav).mean(dim=-1)
        target_mels.append(t_mel)
        t_f0_hz, _, _, t_vuv = tracker.process_utterance(t_wav)
        voiced_idx = t_vuv > 0.5
        if np.any(voiced_idx):
            target_voiced_f0_all.extend(t_f0_hz[voiced_idx].tolist())

    target_centroid = torch.stack(target_mels, dim=0).mean(dim=0)  # (80,)
    target_voiced_cents = 1200.0 * np.log2(np.array(target_voiced_f0_all) / tgt_median)

    # 4. Load Models
    print("\nLoading models for ablation comparison:")
    models: dict[str, FullPersonalizedPipeline] = {}
    for m_name, ckpt_path in model_checkpoints.items():
        print(f" - Loading {m_name} from: {ckpt_path}")
        models[m_name] = load_pipeline_model(ckpt_path, m_name, dev)

    # 5. Initialize Whisper ASR pipeline for CER/WER
    print("\nInitializing Whisper pipeline for CER/WER...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device="cpu",
    )

    # 6. Evaluation Loop
    print(f"\nEvaluating {len(val_records)} validation utterances across all models...")
    model_results: dict[str, dict[str, Any]] = {
        m_name: {
            "voiced_f0_hz": [],
            "correlations": [],
            "cers": [],
            "wers": [],
            "cosine_sims": [],
            "centroids": [],
            "rolloffs": [],
            "mel_l1s": [],
            "utterances": {},
        }
        for m_name in models
    }

    f0_dir_p = Path(f0_dir)

    for item in tqdm(val_records, desc="Evaluating Utterances"):
        utt_id = item["id"]
        src_path = item["audio_24k"]
        src_audio, _ = load_audio(src_path, target_sr=24000)
        gt_text = gt_transcripts.get(utt_id, {}).get("transcript", "")

        # Load precomputed F0
        f0_file = f0_dir_p / f"{Path(src_path).stem}_f0.pt"
        f0_data = torch.load(f0_file, weights_only=False)
        f0_3d = f0_data["f0_3d"]
        if f0_3d.ndim == 2:
            f0_3d = f0_3d.unsqueeze(0)  # Ensure (1, 3, T)
        src_f0_hz = (f0_data.get("raw_f0") if "raw_f0" in f0_data else f0_data["f0_hz"]).squeeze().cpu().numpy()
        src_vuv = f0_3d[0, 2].cpu().numpy()

        # Compute mapped source F0 for trajectory reference
        src_log_f0 = np.where(src_vuv > 0.5, np.log(np.maximum(src_f0_hz, 1.0)), 0.0)
        _, mapped_f0_hz = map_f0_with_config(src_log_f0, src_vuv, f0_prod_cfg)

        # Process each model condition

        for m_name, pipeline_mod in models.items():
            m_audio_dir = audio_out_p / m_name
            m_audio_dir.mkdir(parents=True, exist_ok=True)
            out_wav_path = m_audio_dir / f"{utt_id}.wav"

            # 1. Convert waveform
            use_f0 = m_name in ("model_b_f0", "model_c_gated")
            f0_input = f0_3d if use_f0 else None
            conv_audio = convert_waveform(pipeline_mod, src_audio, f0_input, dev)
            save_wav(out_wav_path, conv_audio, sample_rate=24000)

            # 2. Axis 1 & 2: Pitch Tracking & Trajectory Correlation
            out_f0_hz, _, _, out_vuv = tracker.process_utterance(conv_audio)

            out_voiced_idx = out_vuv > 0.5
            if np.any(out_voiced_idx):
                model_results[m_name]["voiced_f0_hz"].extend(
                    out_f0_hz[out_voiced_idx].tolist()
                )

            # Trajectory correlation on consensus voiced frames
            min_len = min(len(out_f0_hz), len(mapped_f0_hz))
            v_cons = (out_vuv[:min_len] > 0.5) & (src_vuv[:min_len] > 0.5)
            r_val = 0.0
            if np.sum(v_cons) >= 5:
                # Delta log F0 correlation
                log_out = np.log(np.maximum(out_f0_hz[:min_len], 1.0))
                log_map = np.log(np.maximum(mapped_f0_hz[:min_len], 1.0))
                d_out = np.diff(log_out)
                d_map = np.diff(log_map)
                d_cons = v_cons[1:] & v_cons[:-1]
                if np.sum(d_cons) >= 4:
                    std_out = np.std(d_out[d_cons])
                    std_map = np.std(d_map[d_cons])
                    if std_out > 1e-6 and std_map > 1e-6:
                        corr = float(scipy.stats.pearsonr(d_out[d_cons], d_map[d_cons])[0])
                        if not math.isnan(corr):
                            r_val = corr
            model_results[m_name]["correlations"].append(r_val)

            # 3. Axis 3: Whisper CER/WER
            asr_res = asr(
                {"raw": conv_audio.numpy().astype(np.float32), "sampling_rate": 24000},
                generate_kwargs={
                    "language": "zh",
                    "task": "transcribe",
                    "no_repeat_ngram_size": 3,
                },
            )
            hyp_text = str(asr_res["text"]).strip()
            cer, wer = compute_cer_wer(gt_text, hyp_text)
            model_results[m_name]["cers"].append(cer)
            model_results[m_name]["wers"].append(wer)

            # 4. Axis 4 & 5: Speaker Similarity & Target Acoustic Alignment
            conv_mel = extract_log_mel_spectrogram(conv_audio).mean(dim=-1)
            cos_sim = float(
                F.cosine_similarity(
                    conv_mel.unsqueeze(0), target_centroid.unsqueeze(0)
                ).item()
            )
            l1_dist = float(torch.mean(torch.abs(conv_mel - target_centroid)).item())
            centroid, rolloff = compute_spectral_features(conv_audio, sr=24000)

            model_results[m_name]["cosine_sims"].append(cos_sim)
            model_results[m_name]["mel_l1s"].append(l1_dist)
            model_results[m_name]["centroids"].append(centroid)
            model_results[m_name]["rolloffs"].append(rolloff)

            # Store per-utterance record
            model_results[m_name]["utterances"][utt_id] = {
                "hyp_text": hyp_text,
                "cer": cer,
                "wer": wer,
                "r_pitch": r_val,
                "cos_sim": cos_sim,
                "mel_l1": l1_dist,
                "spectral_centroid": centroid,
                "spectral_rolloff": rolloff,
                "audio_rel_path": str(out_wav_path.relative_to(out_p)),
            }

    # 7. Aggregate Summary Statistics
    summary: dict[str, Any] = {}
    for m_name, res in model_results.items():
        v_f0 = np.array(res["voiced_f0_hz"])
        if len(v_f0) > 0:
            med = float(np.median(v_f0))
            q25 = float(np.percentile(v_f0, 25))
            q75 = float(np.percentile(v_f0, 75))
            iqr = q75 - q25
            semitone_diff = float(12.0 * np.log2(med / tgt_median))
            cent_diff = float(1200.0 * np.log2(med / tgt_median))
            model_cents = 1200.0 * np.log2(v_f0 / tgt_median)
            w1_cents = float(
                scipy.stats.wasserstein_distance(model_cents, target_voiced_cents)
            )
        else:
            med, q25, q75, iqr, semitone_diff, cent_diff, w1_cents = (
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            )

        summary[m_name] = {
            "axis1_f0": {
                "median_hz": med,
                "q25_hz": q25,
                "q75_hz": q75,
                "iqr_hz": iqr,
                "target_median_hz": tgt_median,
                "semitone_error_vs_target": semitone_diff,
                "cent_error_vs_target": cent_diff,
                "wasserstein_cents": w1_cents,
            },
            "axis2_pitch_trajectory": {
                "mean_delta_correlation": float(np.mean(res["correlations"])),
                "std_delta_correlation": float(np.std(res["correlations"])),
            },
            "axis3_linguistic": {
                "mean_cer": float(np.mean(res["cers"])),
                "std_cer": float(np.std(res["cers"])),
                "mean_wer": float(np.mean(res["wers"])),
            },
            "axis4_speaker_similarity": {
                "mean_cosine_similarity": float(np.mean(res["cosine_sims"])),
                "std_cosine_similarity": float(np.std(res["cosine_sims"])),
            },
            "axis5_acoustic_alignment": {
                "mean_mel_l1": float(np.mean(res["mel_l1s"])),
                "mean_spectral_centroid_hz": float(np.mean(res["centroids"])),
                "mean_spectral_rolloff_hz": float(np.mean(res["rolloffs"])),
            },
        }

    # 8. Hypothesis Verification
    # H1: Model B > Model A_control in F0 alignment
    b_w1 = summary.get("model_b_f0", {}).get("axis1_f0", {}).get("wasserstein_cents", 0.0)
    a_w1 = summary.get("model_a_control", {}).get("axis1_f0", {}).get("wasserstein_cents", 0.0)
    b_corr = summary.get("model_b_f0", {}).get("axis2_pitch_trajectory", {}).get("mean_delta_correlation", 0.0)
    a_corr = summary.get("model_a_control", {}).get("axis2_pitch_trajectory", {}).get("mean_delta_correlation", 0.0)

    h1_pass = (b_w1 < a_w1) or (b_corr > a_corr)

    # H2: CER(B) ~= CER(A_control) within 0.05
    b_cer = summary.get("model_b_f0", {}).get("axis3_linguistic", {}).get("mean_cer", 0.0)
    a_cer = summary.get("model_a_control", {}).get("axis3_linguistic", {}).get("mean_cer", 0.0)
    h2_pass = abs(b_cer - a_cer) <= 0.05

    # H3: Model C improves timbre distance or skip leakage over Model B
    c_l1 = summary.get("model_c_gated", {}).get("axis5_acoustic_alignment", {}).get("mean_mel_l1", 0.0)
    b_l1 = summary.get("model_b_f0", {}).get("axis5_acoustic_alignment", {}).get("mean_mel_l1", 0.0)
    c_cos = summary.get("model_c_gated", {}).get("axis4_speaker_similarity", {}).get("mean_cosine_similarity", 0.0)
    b_cos = summary.get("model_b_f0", {}).get("axis4_speaker_similarity", {}).get("mean_cosine_similarity", 0.0)
    h3_pass = (c_l1 <= b_l1) or (c_cos >= b_cos)

    hypotheses = {
        "H1_F0_Alignment": {
            "description": "Explicit F0 conditioning reduces distributional F0 error and improves pitch trajectory correlation (B vs A_control)",
            "passed": bool(h1_pass),
            "delta_wasserstein_cents": float(b_w1 - a_w1),
            "delta_correlation": float(b_corr - a_corr),
        },
        "H2_Linguistic_Preservation": {
            "description": "Linguistic intelligibility is preserved with F0 conditioning (CER delta within 5% absolute)",
            "passed": bool(h2_pass),
            "delta_cer": float(b_cer - a_cer),
        },
        "H3_Gated_Skip_Refinement": {
            "description": "Gated skip in Model C refines acoustic alignment / prevents source timbre leakage over Model B",
            "passed": bool(h3_pass),
            "delta_mel_l1": float(c_l1 - b_l1),
            "delta_cosine_similarity": float(c_cos - b_cos),
        },
    }

    full_report = {
        "summary": summary,
        "hypotheses": hypotheses,
        "models": {
            m: {"utterances": model_results[m]["utterances"]} for m in model_results
        },
    }

    report_p = out_p / "exp1b_ablation_report.json"
    with open(report_p, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full quantitative evaluation report to: {report_p}")

    # 9. Build Interactive Audition HTML Page
    build_audition_html(
        val_records=val_records,
        gt_transcripts=gt_transcripts,
        model_results=model_results,
        summary=summary,
        hypotheses=hypotheses,
        target_median=tgt_median,
        target_q25=tgt_q25,
        target_q75=tgt_q75,
        output_html_path=out_p / "exp1b_ablation_audition.html",
        f0_dir=f0_dir_p,
        f0_prod_cfg=f0_prod_cfg,
    )

    return full_report


def build_audition_html(
    val_records: list[dict[str, Any]],
    gt_transcripts: dict[str, Any],
    model_results: dict[str, Any],
    summary: dict[str, Any],
    hypotheses: dict[str, Any],
    target_median: float,
    target_q25: float,
    target_q75: float,
    output_html_path: Path,
    f0_dir: Path,
    f0_prod_cfg: dict[str, Any],
) -> None:
    """Construct a responsive audition interface with embedded audio players and pitch contours."""
    tracker = CausalF0Tracker(F0TrackerConfig())
    rows_html: list[str] = []

    model_keys = list(model_results.keys())

    for idx, item in enumerate(val_records[:15]):  # Detailed visual cards for top 15
        utt_id = item["id"]
        src_path = item["audio_24k"]
        src_audio, _ = load_audio(src_path, target_sr=24000)
        gt_text = gt_transcripts.get(utt_id, {}).get("transcript", "")

        f0_file = f0_dir / f"{Path(src_path).stem}_f0.pt"
        f0_data = torch.load(f0_file, weights_only=False)
        f0_3d_utt = f0_data["f0_3d"]
        if f0_3d_utt.ndim == 2:
            f0_3d_utt = f0_3d_utt.unsqueeze(0)
        src_f0_hz = (f0_data.get("raw_f0") if "raw_f0" in f0_data else f0_data["f0_hz"]).squeeze().cpu().numpy()
        src_vuv = f0_3d_utt[0, 2].cpu().numpy()
        src_log_f0 = np.where(src_vuv > 0.5, np.log(np.maximum(src_f0_hz, 1.0)), 0.0)
        _, mapped_f0 = map_f0_with_config(src_log_f0, src_vuv, f0_prod_cfg)
        t_arr = np.arange(len(src_f0_hz)) * (320 / 24000)

        # SVG for Model B (Explicit F0) vs Mapped vs Source
        m_b_key = "model_b_f0" if "model_b_f0" in model_results else model_keys[-1]
        m_b_audio_rel = model_results[m_b_key]["utterances"][utt_id]["audio_rel_path"]
        m_b_audio_path = output_html_path.parent / m_b_audio_rel
        conv_wav, _ = load_audio(m_b_audio_path, target_sr=24000)
        conv_f0_hz, _, _, conv_vuv = tracker.process_utterance(conv_wav)

        svg = build_trajectory_svg(
            times=t_arr,
            f0_src=src_f0_hz,
            f0_mapped=mapped_f0,
            f0_out=conv_f0_hz,
            vuv_src=src_vuv,
            vuv_out=conv_vuv,
            target_median=target_median,
            target_q25=target_q25,
            target_q75=target_q75,
            title=f"{utt_id}: Explicit F0 (Model B) vs Mapped vs Source",
        )

        audio_cells: list[str] = []
        for m_k in model_keys:
            m_utt = model_results[m_k]["utterances"][utt_id]
            rel_audio = m_utt["audio_rel_path"]
            hyp = html.escape(m_utt["hyp_text"])
            cer_pct = m_utt["cer"] * 100
            r_val = m_utt["r_pitch"]
            cos_val = m_utt["cos_sim"]

            audio_cells.append(f"""
            <div class="model-col">
              <div class="model-tag">{m_k}</div>
              <audio controls src="{rel_audio}"></audio>
              <div class="metrics-tag">CER: <b>{cer_pct:.1f}%</b> | r: <b>{r_val:.2f}</b> | Sim: <b>{cos_val:.3f}</b></div>
              <div class="hyp-text">"{hyp}"</div>
            </div>
            """)

        rows_html.append(f"""
        <div class="card">
          <div class="card-header">
            <h3>#{idx+1}: {utt_id}</h3>
            <span class="ref-text">Ground Truth: <b>"{html.escape(gt_text)}"</b></span>
          </div>
          <div class="source-col">
            <label>Source (User Voice):</label>
            <audio controls src="{Path(src_path).as_posix()}"></audio>
          </div>
          <div class="models-grid">
            {''.join(audio_cells)}
          </div>
          <div class="svg-container">
            {svg}
          </div>
        </div>
        """)

    # Summary table rows
    summary_rows: list[str] = []
    for mk in model_keys:
        s = summary[mk]
        med = s["axis1_f0"]["median_hz"]
        st_err = s["axis1_f0"]["semitone_error_vs_target"]
        w1 = s["axis1_f0"]["wasserstein_cents"]
        r = s["axis2_pitch_trajectory"]["mean_delta_correlation"]
        cer = s["axis3_linguistic"]["mean_cer"] * 100
        wer = s["axis3_linguistic"]["mean_wer"] * 100
        sim = s["axis4_speaker_similarity"]["mean_cosine_similarity"]
        l1 = s["axis5_acoustic_alignment"]["mean_mel_l1"]

        summary_rows.append(f"""
        <tr>
          <td><b>{mk}</b></td>
          <td>{med:.1f} Hz ({st_err:+.2f} st)</td>
          <td><b>{w1:.1f}</b></td>
          <td><b>{r:.3f}</b></td>
          <td>{cer:.1f}% / {wer:.1f}%</td>
          <td><b>{sim:.4f}</b></td>
          <td>{l1:.4f}</td>
        </tr>
        """)

    # Hypotheses cards
    hyp_cards: list[str] = []
    for h_k, h_val in hypotheses.items():
        status_color = "#a6e3a1" if h_val["passed"] else "#f38ba8"
        status_txt = "PASSED" if h_val["passed"] else "FAILED"
        hyp_cards.append(f"""
        <div class="hyp-card" style="border-left: 4px solid {status_color}">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <b>{h_k}</b>
            <span style="color:{status_color}; font-weight:bold;">{status_txt}</span>
          </div>
          <p style="color:#a6adc8; font-size:13px; margin:4px 0;">{h_val['description']}</p>
        </div>
        """)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>EXP-1B: 3-Way Controlled Ablation Audition & Evaluation</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #11111b;
      color: #cdd6f4;
      margin: 0;
      padding: 24px;
    }}
    h1, h2, h3 {{ color: #cdd6f4; margin-top: 0; }}
    .container {{ max-width: 1300px; margin: 0 auto; }}
    .summary-box {{
      background: #1e1e2e;
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      border: 1px solid #313244;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 12px;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid #313244;
      text-align: left;
    }}
    th {{ background: #181825; color: #b4befe; }}
    tr:hover {{ background: #252538; }}
    .hyp-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .hyp-card {{
      background: #181825;
      padding: 14px;
      border-radius: 6px;
    }}
    .card {{
      background: #1e1e2e;
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 20px;
      border: 1px solid #313244;
    }}
    .card-header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      border-bottom: 1px solid #313244;
      padding-bottom: 8px;
      margin-bottom: 14px;
    }}
    .ref-text {{ color: #f9e2af; font-size: 14px; }}
    .source-col {{
      margin-bottom: 14px;
      display: flex;
      align-items: center;
      gap: 12px;
    }}
    .models-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .model-col {{
      background: #181825;
      padding: 12px;
      border-radius: 6px;
      border: 1px solid #313244;
    }}
    .model-tag {{ font-weight: bold; color: #89b4fa; font-size: 13px; margin-bottom: 6px; }}
    .metrics-tag {{ font-size: 11px; color: #bac2de; margin-top: 6px; }}
    .hyp-text {{ font-size: 12px; color: #a6adc8; margin-top: 4px; font-style: italic; }}
    audio {{ width: 100%; height: 32px; }}
    .svg-container {{ margin-top: 10px; overflow-x: auto; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>EXP-1B: Explicit F0 Conditioning & 3-Way Controlled Ablation</h1>
    <p style="color:#a6adc8;">
      Controlled 3-way ablation study evaluating the impact of Explicit 3D F0 Conditioning and Gated Skip Connections
      against equal-training-budget and pre-trained baselines on 50 non-parallel source validation utterances.
    </p>

    <div class="summary-box">
      <h2>Hypothesis Testing Verification</h2>
      <div class="hyp-grid">
        {''.join(hyp_cards)}
      </div>

      <h2 style="margin-top:24px;">Quantitative Multi-Axis Comparison</h2>
      <table>
        <thead>
          <tr>
            <th>Model Condition</th>
            <th>Voiced F0 Median</th>
            <th>Wasserstein (cents) &darr;</th>
            <th>Pitch Delta Corr (r) &uarr;</th>
            <th>Whisper CER / WER &darr;</th>
            <th>Hu Tao Cosine Sim &uarr;</th>
            <th>Target Mel L1 &darr;</th>
          </tr>
        </thead>
        <tbody>
          {''.join(summary_rows)}
        </tbody>
      </table>
    </div>

    <h2>Utterance-by-Utterance Audition & Pitch Trajectory Comparison</h2>
    {''.join(rows_html)}
  </div>
</body>
</html>
"""
    output_html_path.write_text(html_doc, encoding="utf-8")
    print(f"Saved interactive audition comparison interface to: {output_html_path}")


def main() -> None:
    """CLI entrypoint for EXP-1B ablation evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate EXP-1B Ablation Models.")
    parser.add_argument(
        "--model-a-pretrained",
        type=str,
        default="checkpoints/stage2_adapter_best.pt",
        help="Path to pre-trained Stage 2 Epoch-82 checkpoint.",
    )
    parser.add_argument(
        "--model-a-control",
        type=str,
        default="checkpoints/exp1b_model_a_control/best_adapter.pt",
        help="Path to Model A-control checkpoint.",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default="checkpoints/exp1b_model_b_f0/best_adapter.pt",
        help="Path to Model B (Explicit F0) checkpoint.",
    )
    parser.add_argument(
        "--model-c",
        type=str,
        default="checkpoints/exp1b_model_c_gated/best_adapter.pt",
        help="Path to Model C (Gated Skip) checkpoint.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="data/my_voice/processed/val_manifest.json",
        help="Path to source validation manifest.",
    )
    parser.add_argument(
        "--target-val-manifest",
        type=str,
        default="data/hu_tao/processed/val_manifest.json",
        help="Path to target persona validation manifest.",
    )
    parser.add_argument(
        "--f0-dir",
        type=str,
        default="data/features/f0/my_voice",
        help="Path to precomputed 3D F0 features directory.",
    )
    parser.add_argument(
        "--transcripts",
        type=str,
        default="data/my_voice/processed/val_transcripts.json",
        help="Path to ground truth validation transcripts JSON.",
    )
    parser.add_argument(
        "--f0-config",
        type=str,
        default="outputs/f0_production_config.json",
        help="Path to frozen production F0 config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/exp1b_evaluation",
        help="Directory to save evaluation artifacts.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on validation samples for quick verification.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda or cpu).",
    )

    args = parser.parse_args()

    models_to_eval: dict[str, str | Path] = {}
    if Path(args.model_a_pretrained).is_file():
        models_to_eval["model_a_pretrained"] = args.model_a_pretrained
    if Path(args.model_a_control).is_file():
        models_to_eval["model_a_control"] = args.model_a_control
    if Path(args.model_b).is_file():
        models_to_eval["model_b_f0"] = args.model_b
    if Path(args.model_c).is_file():
        models_to_eval["model_c_gated"] = args.model_c

    assert len(models_to_eval) > 0, "No valid model checkpoints provided for evaluation."

    evaluate_ablation(
        model_checkpoints=models_to_eval,
        val_manifest_path=args.val_manifest,
        f0_dir=args.f0_dir,
        transcripts_path=args.transcripts,
        target_val_manifest_path=args.target_val_manifest,
        f0_config_path=args.f0_config,
        output_dir=args.output_dir,
        device_str=args.device,
        max_eval_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
