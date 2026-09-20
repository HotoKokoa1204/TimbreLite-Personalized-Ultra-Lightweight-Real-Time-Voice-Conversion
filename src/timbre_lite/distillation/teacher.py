"""Offline ContentVec phonetic teacher feature extraction and caching engine."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import HubertModel

from timbre_lite.data.loader import load_audio
from timbre_lite.distillation.alignment import align_teacher_features


class ContentVecTeacher(nn.Module):
    """Offline phonetic teacher model using ContentVec (lengyue233/content-vec-best).

    Extracts 768-dimensional phonetic speech representations from 16kHz audio.
    Runs strictly offline during preprocessing and is completely excluded from
    the live streaming inference path.
    """

    def __init__(
        self,
        model_id: str = "lengyue233/content-vec-best",
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize ContentVec teacher model.

        Hardware selection priority: CUDA GPU acceleration when available,
        with automatic CPU fallback.

        Args:
            model_id: Hugging Face repository identifier for ContentVec checkpoint.
            device: Optional explicit torch device. If None, selects CUDA if
                available, else CPU.
        """
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_id = model_id
        # Load Hubert architecture checkpoint
        self.model = HubertModel.from_pretrained(model_id)
        self.model.eval()
        self.model.to(self.device)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def extract_features(self, audio_16k: torch.Tensor) -> torch.Tensor:
        """Extract 768-d phonetic teacher features from 16kHz audio waveform.

        Args:
            audio_16k: 1D tensor of shape (samples,) or 2D of shape (batch, samples).

        Returns:
            Feature tensor of shape (batch, 768, time_50hz) at 50Hz frame rate.
        """
        if audio_16k.ndim == 1:
            audio_16k = audio_16k.unsqueeze(0)

        audio_input = audio_16k.to(device=self.device, dtype=torch.float32)

        # Forward pass through HuBERT encoder
        outputs = self.model(input_values=audio_input)
        last_hidden: torch.Tensor = outputs.last_hidden_state  # (batch, time, 768)

        # Transpose to (batch, channels=768, time) for 1D convolutions
        features = last_hidden.transpose(1, 2)
        return features

    @torch.no_grad()
    def extract_and_cache_manifest(
        self,
        manifest_path: str | Path,
        cache_dir: str | Path,
        align_to_codec_frames: bool = True,
        codec_hop: int = 320,
    ) -> list[dict[str, object]]:
        """Extract teacher features for all records in a manifest and cache to disk.

        Args:
            manifest_path: Path to input manifest JSON file.
            cache_dir: Directory to write cached .pt feature tensors.
            align_to_codec_frames: If True, temporally interpolate features
                to match EnCodec 75Hz frame count.
            codec_hop: EnCodec frame hop in samples at 24kHz (default 320).

        Returns:
            Updated manifest records containing 'teacher_features_path' and
            'num_feature_frames'.
        """
        in_path = Path(manifest_path)
        out_dir = Path(cache_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(in_path, encoding="utf-8") as f:
            records: list[dict[str, object]] = json.load(f)

        updated_records: list[dict[str, object]] = []

        for item in tqdm(records, desc=f"Extracting ContentVec ({in_path.stem})"):
            audio_16k_path = str(item["audio_16k"])
            item_id = str(item["id"])
            target_pt = out_dir / f"{item_id}_contentvec.pt"

            if target_pt.is_file():
                # Already cached
                cached = torch.load(target_pt, weights_only=True)
                item["teacher_features_path"] = str(target_pt)
                item["num_feature_frames"] = cached.shape[-1]
                updated_records.append(item)
                continue

            # Load 16k audio
            audio_16k, _ = load_audio(audio_16k_path, target_sr=16000)
            feat_50hz = self.extract_features(audio_16k)  # (1, 768, T_50)

            if align_to_codec_frames:
                num_samples_24k = int(item["num_samples_24k"])
                target_frames = max(1, num_samples_24k // codec_hop)
                feat = align_teacher_features(feat_50hz, target_frames=target_frames)
            else:
                feat = feat_50hz

            feat_cpu = feat.squeeze(0).cpu().to(dtype=torch.float32)
            torch.save(feat_cpu, target_pt)

            item["teacher_features_path"] = str(target_pt)
            item["num_feature_frames"] = feat_cpu.shape[-1]
            updated_records.append(item)

        # Overwrite manifest with cached paths
        with open(in_path, "w", encoding="utf-8") as f:
            json.dump(updated_records, f, indent=2, ensure_ascii=False)

        return updated_records
