"""Ingestion, downloading, and standardization of Hu Tao target persona voice data."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as taf
from tqdm import tqdm

from timbre_lite.data.loader import load_audio, save_wav


@dataclass
class HuTaoDatasetConfig:
    """Configuration parameters for Hu Tao target dataset curation.

    Attributes:
        repo_id: Hugging Face dataset repository identifier.
        zip_filename: Archive path inside the Hugging Face repository.
        codec_sr: Sampling rate for EnCodec streaming (24kHz).
        teacher_sr: Sampling rate for ContentVec teacher extraction (16kHz).
        target_loudness_dbfs: Target loudness in dBFS (-20 dBFS).
        train_ratio: Fraction of dataset allocated to training manifest.
        seed: Deterministic random seed for train/val split.
    """

    repo_id: str = "simon3000/genshin-voice"
    zip_filename: str = "speaker-archives/Chinese/501-plus/Hu_Tao.zip"
    codec_sr: int = 24000
    teacher_sr: int = 16000
    target_loudness_dbfs: float = -20.0
    train_ratio: float = 0.9
    seed: int = 42


def download_hu_tao_dataset(
    target_zip_path: str | Path,
    config: HuTaoDatasetConfig | None = None,
) -> Path:
    """Download Hu Tao voice archive from Hugging Face if not already present.

    Args:
        target_zip_path: Local filesystem path where Hu_Tao.zip should reside.
        config: Optional HuTaoDatasetConfig override.

    Returns:
        Path to the downloaded zip archive.
    """
    cfg = config or HuTaoDatasetConfig()
    zip_path = Path(target_zip_path)
    if zip_path.is_file() and zip_path.stat().st_size > 1024 * 1024:
        return zip_path

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    # Download from Hugging Face with automatic progress bar
    downloaded_path = hf_hub_download(
        repo_id=cfg.repo_id,
        repo_type="dataset",
        filename=cfg.zip_filename,
        local_dir=str(zip_path.parent.parent),
    )
    downloaded = Path(downloaded_path)
    if downloaded != zip_path and downloaded.is_file():
        import shutil

        shutil.move(str(downloaded), str(zip_path))

    return zip_path


def curate_hu_tao_dataset(
    zip_path: str | Path,
    output_dir: str | Path,
    config: HuTaoDatasetConfig | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Extract, resample, normalize, and manifest Hu Tao studio voice lines.

    Standardizes all 521+ clean studio voice lines into 24kHz (EnCodec) and
    16kHz (ContentVec) mono WAVs normalized to -20 dBFS.

    Args:
        zip_path: Path to downloaded Hu_Tao.zip archive.
        output_dir: Base output directory for processed files and manifests.
        config: Optional HuTaoDatasetConfig override.

    Returns:
        Tuple of (train_manifest_records, val_manifest_records).
    """
    cfg = config or HuTaoDatasetConfig()
    out_base = Path(output_dir)
    raw_dir = out_base / "raw"
    dir_24k = out_base / "24k"
    dir_16k = out_base / "16k"
    raw_dir.mkdir(parents=True, exist_ok=True)
    dir_24k.mkdir(parents=True, exist_ok=True)
    dir_16k.mkdir(parents=True, exist_ok=True)

    # 1. Unzip archive if raw directory is empty
    archive_path = Path(zip_path)
    wav_files = list(raw_dir.glob("*.wav")) + list(raw_dir.glob("**/*.wav"))
    if not wav_files:
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(raw_dir)
        wav_files = list(raw_dir.glob("*.wav")) + list(raw_dir.glob("**/*.wav"))

    target_rms = 10.0 ** (cfg.target_loudness_dbfs / 20.0)  # 0.1 at -20 dBFS
    samples: list[dict[str, object]] = []

    for idx, wav_path in enumerate(tqdm(sorted(wav_files), desc="Curating Hu Tao")):
        try:
            audio, sr = load_audio(wav_path)
        except Exception:
            continue

        if len(audio) < int(sr * 0.5):
            # Discard fragments shorter than 0.5s
            continue

        # Resample to 24kHz and 16kHz
        if sr != cfg.codec_sr:
            t_24k = taf.resample(audio, orig_freq=sr, new_freq=cfg.codec_sr)
        else:
            t_24k = audio

        if sr != cfg.teacher_sr:
            t_16k = taf.resample(audio, orig_freq=sr, new_freq=cfg.teacher_sr)
        else:
            t_16k = audio

        # Normalize loudness on 24k audio
        rms = torch.sqrt(torch.mean(t_24k**2)) + 1e-9
        gain = target_rms / float(rms)
        t_24k = t_24k * gain
        t_16k = t_16k * gain

        # Peak limiter
        max_val = max(
            float(torch.max(torch.abs(t_24k))),
            float(torch.max(torch.abs(t_16k))),
        )
        if max_val > 0.99:
            scale = 0.99 / max_val
            t_24k = t_24k * scale
            t_16k = t_16k * scale

        sample_id = f"hutao_utt_{idx + 1:04d}"
        path_24k = dir_24k / f"{sample_id}.wav"
        path_16k = dir_16k / f"{sample_id}.wav"

        save_wav(path_24k, t_24k, cfg.codec_sr)
        save_wav(path_16k, t_16k, cfg.teacher_sr)

        samples.append(
            {
                "id": sample_id,
                "audio_24k": str(path_24k),
                "audio_16k": str(path_16k),
                "duration_sec": round(len(t_24k) / cfg.codec_sr, 3),
                "num_samples_24k": len(t_24k),
                "num_samples_16k": len(t_16k),
            }
        )

    # Train/Val split (90/10)
    rng = np.random.default_rng(cfg.seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)

    split_point = int(len(samples) * cfg.train_ratio)
    train_idx = set(indices[:split_point])

    train_records: list[dict[str, object]] = []
    val_records: list[dict[str, object]] = []

    for i, rec in enumerate(samples):
        if i in train_idx:
            train_records.append(rec)
        else:
            val_records.append(rec)

    with open(out_base / "train_manifest.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)

    with open(out_base / "val_manifest.json", "w", encoding="utf-8") as f:
        json.dump(val_records, f, indent=2, ensure_ascii=False)

    return train_records, val_records


if __name__ == "__main__":
    import sys

    dest_dir = sys.argv[1] if len(sys.argv) > 1 else "data/hu_tao"
    zip_p = Path(dest_dir) / "Hu_Tao.zip"
    print(f"Checking/downloading Hu Tao dataset to {zip_p}...")
    download_hu_tao_dataset(zip_p)
    print(f"Curating Hu Tao dataset into {Path(dest_dir) / 'processed'}...")
    tr, vr = curate_hu_tao_dataset(zip_p, Path(dest_dir) / "processed")
    print(
        f"Hu Tao curation complete! Total: {len(tr) + len(vr)} "
        f"({len(tr)} train, {len(vr)} val)."
    )
