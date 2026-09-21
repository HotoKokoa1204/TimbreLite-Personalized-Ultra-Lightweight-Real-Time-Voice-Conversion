"""Precompute and cache 3D normalized F0 features [n_t, delta_n_t, vuv_t] with metadata."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torchaudio import transforms as ta_transforms

from timbre_lite.modules.f0 import CausalF0Tracker, F0TrackerConfig


def get_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def precompute_f0_for_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    speaker_type: str,
    prod_config_path: str | Path = "outputs/f0_production_config.json",
    sr: int = 24000,
) -> int:
    """Extract and save 3D F0 features with full provenance metadata.

    Args:
        manifest_path: Path to dataset manifest JSON.
        output_dir: Target directory to save .pt tensor files.
        speaker_type: 'source' or 'target'.
        prod_config_path: Path to frozen production F0 configuration.
        sr: Audio sample rate (24000).

    Returns:
        Number of processed audio files.
    """
    m_p = Path(manifest_path)
    assert m_p.exists(), f"Manifest {manifest_path} not found"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_p = Path(prod_config_path)
    assert cfg_p.exists(), f"Production config {prod_config_path} not found"
    config_hash = get_file_hash(cfg_p)
    prod_config = json.loads(cfg_p.read_text(encoding="utf-8"))

    if speaker_type == "target":
        mu = float(prod_config["target"]["median_log_f0"])
        iqr = float(prod_config["target"]["iqr_log_f0"])
        f_min, f_max = 140.0, 800.0
    else:
        mu = float(prod_config["source"]["median_log_f0"])
        iqr = float(prod_config["source"]["iqr_log_f0"])
        f_min, f_max = 65.0, 380.0

    tracker_cfg = F0TrackerConfig(
        sample_rate=sr,
        hop_length=320,
        window_length=1280,
        f_min=f_min,
        f_max=f_max,
    )
    tracker = CausalF0Tracker(tracker_cfg)

    with open(m_p, encoding="utf-8") as f:
        manifest = json.load(f)

    count = 0
    for item in manifest:
        audio_path_str = item.get("audio_24k") or item.get("audio")
        if not audio_path_str:
            continue
        audio_path = Path(audio_path_str)
        if not audio_path.exists():
            continue

        wav, file_sr = sf.read(str(audio_path))
        if file_sr != sr:
            t_data = torch.from_numpy(wav).float().unsqueeze(0)
            resampler = ta_transforms.Resample(orig_freq=file_sr, new_freq=sr)
            wav = resampler(t_data).squeeze(0).numpy()

        f0_traj, log_f0_traj, _, vuv = tracker.process_utterance(wav)
        n_frames = len(vuv)

        # 1. n_t: normalized log-F0
        norm_log_f0 = np.zeros(n_frames, dtype=np.float32)
        v_mask = vuv > 0.5
        norm_log_f0[v_mask] = (log_f0_traj[v_mask] - mu) / max(iqr, 1e-6)

        # 2. delta_n_t: V -> V transitions only
        delta_norm_log_f0 = np.zeros(n_frames, dtype=np.float32)
        for t in range(1, n_frames):
            if vuv[t] > 0.5 and vuv[t - 1] > 0.5:
                diff = norm_log_f0[t] - norm_log_f0[t - 1]
                delta_norm_log_f0[t] = np.clip(diff, -1.0, 1.0)

        # Combine into (3, n_frames) tensor
        feat_3d = np.stack(
            [norm_log_f0, delta_norm_log_f0, vuv.astype(np.float32)], axis=0
        )

        record: dict[str, Any] = {
            "f0_3d": torch.from_numpy(feat_3d).float(),
            "raw_f0": torch.from_numpy(f0_traj).float(),
            "num_frames": n_frames,
            "sample_rate": sr,
            "hop_size": 320,
            "window_size": 1280,
            "speaker_type": speaker_type,
            "f0_extraction_mode": "causal",
            "delta_definition_version": "v1_normalized_v2v",
            "tracker_version": "causal_boersma_v1",
            "f0_config_hash": config_hash,
            "mu_log_f0": mu,
            "iqr_log_f0": iqr,
        }

        save_name = f"{audio_path.stem}_f0.pt"
        torch.save(record, out_dir / save_name)
        count += 1

    return count


def main() -> None:
    """Precompute F0 features across all training and validation datasets."""
    print("=" * 70)
    print("PRECOMPUTING 3D F0 FEATURES FOR EXP-1B")
    print("=" * 70)

    datasets = [
        ("data/hu_tao/processed/train_manifest.json", "data/features/f0/hu_tao", "target"),
        ("data/hu_tao/processed/val_manifest.json", "data/features/f0/hu_tao", "target"),
        ("data/my_voice/processed/train_manifest.json", "data/features/f0/my_voice", "source"),
        ("data/my_voice/processed/val_manifest.json", "data/features/f0/my_voice", "source"),
    ]

    total = 0
    for manifest_path, out_dir, spk_type in datasets:
        p = Path(manifest_path)
        if p.exists():
            c = precompute_f0_for_manifest(manifest_path, out_dir, spk_type)
            print(f"Processed {c:3d} utterances from {manifest_path} -> {out_dir}")
            total += c

    print(f"\nSuccessfully cached {total} 3D F0 feature files.")


if __name__ == "__main__":
    main()
