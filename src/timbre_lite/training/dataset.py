"""Voice conversion dataset with decoupled training crops and caching."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from timbre_lite.data.loader import load_audio


class CropStrategy(str, Enum):
    """Temporal crop strategy during training data batching."""

    RANDOM = "random"
    CENTER = "center"
    NONE = "none"


class VoiceConversionDataset(Dataset[dict[str, Any]]):
    """Dataset loading dual-rate audio and pre-cached phonetic teacher representations.

    Decouples raw VAD utterance segmentation (2.0s - 8.0s) from training time
    window cropping (e.g. 2.56s, 3.2s, 5.12s, or variable length).
    """

    def __init__(
        self,
        manifest_path: str | Path,
        crop_sec: float | None = 2.56,
        crop_strategy: CropStrategy = CropStrategy.RANDOM,
        codec_sr: int = 24000,
        teacher_sr: int = 16000,
        load_features: bool = True,
    ) -> None:
        """Initialize VoiceConversionDataset.

        Args:
            manifest_path: Path to train or val manifest JSON.
            crop_sec: Target duration in seconds for training crops. If None,
                returns full variable-length utterances.
            crop_strategy: RANDOM, CENTER, or NONE.
            codec_sr: Sampling rate for 24kHz codec audio.
            teacher_sr: Sampling rate for 16kHz teacher audio.
            load_features: Whether to attempt loading cached teacher .pt tensors.
        """
        super().__init__()
        self.manifest_path = Path(manifest_path)
        with open(self.manifest_path, encoding="utf-8") as f:
            self.records: list[dict[str, Any]] = json.load(f)

        self.crop_sec = crop_sec
        self.crop_strategy = crop_strategy
        self.codec_sr = codec_sr
        self.teacher_sr = teacher_sr
        self.load_features = load_features

        self.crop_samples_24k: int | None = (
            int(crop_sec * codec_sr) if crop_sec is not None else None
        )
        self.crop_samples_16k: int | None = (
            int(crop_sec * teacher_sr) if crop_sec is not None else None
        )

    def __len__(self) -> int:
        """Return number of utterances in dataset."""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Load and crop a single audio sample and its teacher features.

        Args:
            idx: Sample integer index.

        Returns:
            Dictionary containing 'id', 'audio_24k', 'audio_16k', 'duration_sec',
            and optional 'teacher_features'.
        """
        item = self.records[idx]
        sample_id = str(item["id"])
        audio_24k_path = str(item["audio_24k"])
        audio_16k_path = str(item["audio_16k"])

        # Load audio waveforms
        audio_24k, _ = load_audio(audio_24k_path, target_sr=self.codec_sr)
        audio_16k, _ = load_audio(audio_16k_path, target_sr=self.teacher_sr)

        teacher_features: torch.Tensor | None = None
        if self.load_features and "teacher_features_path" in item:
            feat_path = Path(str(item["teacher_features_path"]))
            if feat_path.is_file():
                teacher_features = torch.load(feat_path, weights_only=True)

        total_samples_24k = len(audio_24k)

        # Apply temporal cropping if enabled and audio is sufficiently long
        if (
            self.crop_samples_24k is not None
            and total_samples_24k > self.crop_samples_24k
        ):
            crop_len_24k = self.crop_samples_24k
            crop_len_16k = self.crop_samples_16k or int(crop_len_24k * (16000 / 24000))
            max_offset_24k = total_samples_24k - crop_len_24k

            if self.crop_strategy == CropStrategy.RANDOM:
                offset_24k = int(np.random.randint(0, max_offset_24k + 1))
            elif self.crop_strategy == CropStrategy.CENTER:
                offset_24k = max_offset_24k // 2
            else:
                offset_24k = 0

            offset_16k = int(round(offset_24k * (self.teacher_sr / self.codec_sr)))

            audio_24k = audio_24k[offset_24k : offset_24k + crop_len_24k]
            audio_16k = audio_16k[offset_16k : offset_16k + crop_len_16k]

            if teacher_features is not None:
                # Teacher features are aligned at 75Hz (hop = 320 samples at 24k)
                offset_frames = offset_24k // 320
                num_frames = crop_len_24k // 320
                teacher_features = teacher_features[
                    :, offset_frames : offset_frames + num_frames
                ]

        result: dict[str, Any] = {
            "id": sample_id,
            "audio_24k": audio_24k,
            "audio_16k": audio_16k,
            "duration_sec": len(audio_24k) / self.codec_sr,
        }

        if teacher_features is not None:
            result["teacher_features"] = teacher_features

        return result


def collate_voice_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length utterances and features into a aligned batch tensor.

    Args:
        batch: List of sample dictionaries returned by VoiceConversionDataset.

    Returns:
        Collate dictionary containing padded 'audio_24k', 'audio_16k',
        optional 'teacher_features', and 'mask'.
    """
    max_len_24k = max(len(b["audio_24k"]) for b in batch)
    max_len_16k = max(len(b["audio_16k"]) for b in batch)
    batch_size = len(batch)

    padded_24k = torch.zeros(batch_size, max_len_24k, dtype=torch.float32)
    padded_16k = torch.zeros(batch_size, max_len_16k, dtype=torch.float32)
    lens_24k = torch.tensor([len(b["audio_24k"]) for b in batch], dtype=torch.long)

    for i, b in enumerate(batch):
        l24 = len(b["audio_24k"])
        l16 = len(b["audio_16k"])
        padded_24k[i, :l24] = b["audio_24k"]
        padded_16k[i, :l16] = b["audio_16k"]

    result: dict[str, Any] = {
        "id": [b["id"] for b in batch],
        "audio_24k": padded_24k,
        "audio_16k": padded_16k,
        "lens_24k": lens_24k,
    }

    # If teacher features are present across batch, pad them
    has_teacher = all("teacher_features" in b for b in batch)
    if has_teacher:
        channels = batch[0]["teacher_features"].shape[0]
        max_frames = max(b["teacher_features"].shape[-1] for b in batch)
        padded_feat = torch.zeros(batch_size, channels, max_frames, dtype=torch.float32)
        mask = torch.zeros(batch_size, max_frames, dtype=torch.float32)

        for i, b in enumerate(batch):
            tf = b["teacher_features"]
            f_len = tf.shape[-1]
            padded_feat[i, :, :f_len] = tf
            mask[i, :f_len] = 1.0

        result["teacher_features"] = padded_feat
        result["mask"] = mask

    return result
