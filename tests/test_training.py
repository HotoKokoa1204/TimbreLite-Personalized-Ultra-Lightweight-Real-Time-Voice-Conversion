"""Unit tests for voice conversion dataset, decoupled crop strategies, and trainers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.data.loader import save_wav
from timbre_lite.distillation.loss import DistillationProjectionHead
from timbre_lite.modules.adapter import FullPersonalizedPipeline
from timbre_lite.modules.cleanser import ContentCleanser
from timbre_lite.training.dataset import (
    CropStrategy,
    VoiceConversionDataset,
    collate_voice_batch,
)
from timbre_lite.training.losses import LatentLoss, MultiScaleSTFTLoss
from timbre_lite.training.trainer import AdapterTrainer, DistillationTrainer


def test_dataset_decoupled_crops_and_collate() -> None:
    """Verify dataset implements 2.56s, 3.2s and variable length crops."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_path = Path(tmp_dir)
        wav24_path = base_path / "sample_24k.wav"
        wav16_path = base_path / "sample_16k.wav"
        feat_path = base_path / "sample_feat.pt"

        # 4 seconds sample utterance (within 2.0s - 8.0s VAD bounds)
        dur = 4.0
        s24 = int(24000 * dur)
        s16 = int(16000 * dur)
        frames75 = s24 // 320  # 300 frames

        save_wav(wav24_path, torch.randn(s24) * 0.1, 24000)
        save_wav(wav16_path, torch.randn(s16) * 0.1, 16000)
        torch.save(torch.randn(768, frames75), feat_path)

        manifest_path = base_path / "test_manifest.json"
        record = [
            {
                "id": "test_utt_01",
                "audio_24k": str(wav24_path),
                "audio_16k": str(wav16_path),
                "teacher_features_path": str(feat_path),
                "duration_sec": dur,
                "num_samples_24k": s24,
                "num_samples_16k": s16,
            }
        ]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(record, f)

        # 1. Test 2.56s random crop
        ds_256 = VoiceConversionDataset(
            manifest_path=manifest_path,
            crop_sec=2.56,
            crop_strategy=CropStrategy.RANDOM,
        )
        sample_256 = ds_256[0]
        assert len(sample_256["audio_24k"]) == int(2.56 * 24000)
        assert len(sample_256["audio_16k"]) == int(2.56 * 16000)
        assert sample_256["teacher_features"].shape[-1] == int(2.56 * 24000) // 320

        # 2. Test 3.2s center crop
        ds_320 = VoiceConversionDataset(
            manifest_path=manifest_path,
            crop_sec=3.2,
            crop_strategy=CropStrategy.CENTER,
        )
        sample_320 = ds_320[0]
        assert len(sample_320["audio_24k"]) == int(3.2 * 24000)
        assert len(sample_320["audio_16k"]) == int(3.2 * 16000)

        # 3. Test Full Utterance (no crop)
        ds_full = VoiceConversionDataset(
            manifest_path=manifest_path,
            crop_sec=None,
            crop_strategy=CropStrategy.NONE,
        )
        sample_full = ds_full[0]
        assert len(sample_full["audio_24k"]) == s24

        # 4. Test Collate function
        batch = collate_voice_batch([sample_256, sample_320])
        assert batch["audio_24k"].shape[0] == 2
        assert batch["audio_24k"].shape[1] == int(3.2 * 24000)  # Max length
        assert "mask" in batch


def test_losses_multiscale_stft_and_latent() -> None:
    """Verify MultiScaleSTFTLoss and LatentLoss computation."""
    stft_loss = MultiScaleSTFTLoss(
        fft_sizes=(512, 1024),
        hop_sizes=(50, 120),
        win_lengths=(240, 600),
    )
    # 0.5s audio at 24kHz = 12000 samples
    x = torch.randn(2, 12000)
    y = x + (torch.randn(2, 12000) * 0.05)
    loss = stft_loss(x, y)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    latent_loss = LatentLoss()
    z1 = torch.randn(2, 128, 50)
    z2 = torch.randn(2, 128, 50)
    l_lat = latent_loss(z1, z2)
    assert l_lat.item() > 0.0


def test_distillation_trainer_step_and_freezing() -> None:
    """Verify DistillationTrainer step updates cleanser while EnCodec remains frozen."""
    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    proj_head = DistillationProjectionHead(in_dim=64, out_dim=768)

    trainer = DistillationTrainer(
        cleanser=cleanser,
        proj_head=proj_head,
        codec=codec,
        device="cpu",
    )

    # 1 second of 24kHz audio = 24000 samples -> 75 frames
    audio_24k = torch.randn(2, 24000)
    teacher_feat = torch.randn(2, 768, 75)

    metrics = trainer.train_step(audio_24k, teacher_feat)
    assert "loss_distill_total" in metrics
    assert metrics["loss_distill_total"] > 0.0

    # Assert EnCodec parameters remained frozen
    for p in codec.model.parameters():
        assert p.grad is None


def test_adapter_trainer_step() -> None:
    """Verify AdapterTrainer forward-backward optimization step."""
    from timbre_lite.modules.adapter import DualStreamFusion, PersonalizedAdapter
    from timbre_lite.modules.prosody import InGraphProsodyHead

    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    prosody = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)
    adapter = PersonalizedAdapter(in_dim=64, out_dim=128)
    pipeline = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody,
        fusion=fusion,
        adapter=adapter,
    )
    trainer = AdapterTrainer(pipeline=pipeline, device="cpu")

    # Target audio batch
    target_audio = torch.randn(1, 24000)
    metrics = trainer.train_step(target_audio)

    assert "loss_adapter_total" in metrics
    assert metrics["loss_adapter_total"] > 0.0


def test_stage2_resume_from_checkpoint() -> None:
    """Verify Stage 2 training can resume state from a saved checkpoint."""
    from timbre_lite.training.train_stage2 import train_stage2

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        wav_path = tmp_path / "target.wav"
        manifest_path = tmp_path / "manifest.json"
        ckpt_dir = tmp_path / "checkpoints"

        # Create 3 seconds of dummy audio
        save_wav(wav_path, torch.randn(24000 * 3) * 0.1, 24000)
        manifest_data = [
            {
                "id": "target_01",
                "audio_24k": str(wav_path),
                "audio_16k": str(wav_path),
                "duration_sec": 3.0,
            }
        ]
        manifest_path.write_text(json.dumps(manifest_data), encoding="utf-8")

        # 1. Run for 1 epoch
        best_1 = train_stage2(
            target_manifest=manifest_path,
            epochs=1,
            batch_size=1,
            checkpoint_dir=ckpt_dir,
            device="cpu",
        )
        assert best_1.is_file()
        ckpt_1 = torch.load(ckpt_dir / "stage2_adapter_latest.pt", weights_only=False)
        assert ckpt_1["epoch"] == 1

        # 2. Resume to epoch 2
        best_2 = train_stage2(
            target_manifest=manifest_path,
            epochs=2,
            batch_size=1,
            checkpoint_dir=ckpt_dir,
            resume_from=ckpt_dir / "stage2_adapter_latest.pt",
            device="cpu",
        )
        assert best_2.is_file()
        ckpt_2 = torch.load(ckpt_dir / "stage2_adapter_latest.pt", weights_only=False)
        assert ckpt_2["epoch"] == 2
