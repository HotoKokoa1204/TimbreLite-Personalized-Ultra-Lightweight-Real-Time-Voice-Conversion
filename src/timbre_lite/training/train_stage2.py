"""Stage 2: Target persona timbre manifold auto-reconstruction training script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.modules.adapter import (
    DualStreamFusion,
    FullPersonalizedPipeline,
    InGraphProsodyHead,
    PersonalizedAdapter,
)
from timbre_lite.modules.cleanser import ContentCleanser
from timbre_lite.training.dataset import (
    CropStrategy,
    VoiceConversionDataset,
    collate_voice_batch,
)
from timbre_lite.training.trainer import AdapterTrainer


def train_stage2(
    target_manifest: str | Path,
    val_manifest: str | Path | None = None,
    stage1_checkpoint: str | Path | None = None,
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 2e-4,
    crop_sec: float = 2.56,
    checkpoint_dir: str | Path = "checkpoints",
    device: torch.device | str | None = None,
) -> Path:
    """Run Stage 2 target manifold auto-reconstruction training loop.

    Args:
        target_manifest: Path to target speaker training manifest.
        val_manifest: Optional path to target speaker validation manifest.
        stage1_checkpoint: Optional path to pretrained Stage 1 Content Cleanser.
        epochs: Number of training epochs (default: 100).
        batch_size: Mini-batch size.
        lr: Learning rate for AdamW optimizer.
        crop_sec: Training crop window duration in seconds.
        checkpoint_dir: Output checkpoint directory.
        device: Target compute device.

    Returns:
        Path to best saved Stage 2 checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Stabilize cuDNN to prevent CUDNN_STATUS_INTERNAL_ERROR on long runs.
    # benchmark=True causes cuDNN to profile & pick fastest kernels each batch,
    # which grows a fragmented workspace pool over 80+ epochs and eventually
    # fails to allocate during backward(). Setting both flags off forces a
    # single deterministic workspace path with bounded memory usage.
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

    print(f"Initializing Stage 2 training ({epochs} epochs) on device: {dev}")

    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)

    # Load Stage 1 cleanser if provided
    if stage1_checkpoint is not None and Path(stage1_checkpoint).is_file():
        print(f"Loading Stage 1 Content Cleanser from {stage1_checkpoint}...")
        ckpt = torch.load(stage1_checkpoint, map_location="cpu", weights_only=False)
        if "cleanser_state_dict" in ckpt:
            cleanser.load_state_dict(ckpt["cleanser_state_dict"])
        elif isinstance(ckpt, dict):
            cleanser.load_state_dict(ckpt)
    else:
        print("Note: Stage 1 checkpoint not provided or not found. Using raw Cleanser.")

    # Content Cleanser remains frozen during Stage 2
    for param in cleanser.parameters():
        param.requires_grad = False
    cleanser.eval()

    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)
    adapter = PersonalizedAdapter(
        in_dim=64, out_dim=128, hidden_dim=64, tcn_layers=4, gru_hidden=64
    )

    pipeline = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter,
    ).to(dev)

    trainer = AdapterTrainer(pipeline=pipeline, lr=lr, device=dev)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs, eta_min=1e-5
    )

    train_ds = VoiceConversionDataset(
        manifest_path=target_manifest,
        crop_sec=crop_sec,
        crop_strategy=CropStrategy.RANDOM,
        load_features=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_voice_batch,
    )

    val_loader: DataLoader[dict[str, Any]] | None = None
    if val_manifest is not None and Path(val_manifest).is_file():
        val_ds = VoiceConversionDataset(
            manifest_path=val_manifest,
            crop_sec=None,
            crop_strategy=CropStrategy.NONE,
            load_features=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_voice_batch,
        )

    best_loss = float("inf")
    best_ckpt_path = ckpt_dir / "stage2_adapter_best.pt"
    history: list[dict[str, float]] = []

    epoch_pbar = tqdm(
        range(1, epochs + 1),
        desc="Stage 2 Progress (100 Epochs)",
        unit="epoch",
        dynamic_ncols=True,
    )

    for epoch in epoch_pbar:
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:03d}/{epochs:03d}",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            metrics = trainer.train_step(
                target_audio_24k=batch["audio_24k"],
                mask=batch.get("mask"),
            )
            step_loss = metrics.get("loss_adapter_total", 0.0)
            total_loss += step_loss
            num_batches += 1
            pbar.set_postfix(
                {
                    "total": f"{step_loss:.3f}",
                    "stft": f"{metrics.get('loss_adapter_stft', 0.0):.3f}",
                    "lat": f"{metrics.get('loss_adapter_latent', 0.0):.3f}",
                }
            )

        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation loop
        avg_val_loss = avg_train_loss
        if val_loader is not None:
            val_loss_sum = 0.0
            val_count = 0
            pipeline.eval()
            codec_model: Any = pipeline.codec.model
            with torch.no_grad():
                for v_batch in val_loader:
                    audio = v_batch["audio_24k"].to(dev)
                    if audio.ndim == 2:
                        audio_in = audio.unsqueeze(1)
                    else:
                        audio_in = audio
                    z_target = codec_model.encoder(audio_in)
                    z_adapted, _, _ = pipeline.forward_sequence(z_target)
                    y_recon = codec_model.decoder(z_adapted).squeeze(1)

                    loss_stft = trainer.stft_loss(y_recon, audio)
                    loss_lat = trainer.latent_loss(z_adapted, z_target)
                    val_step = loss_stft + (2.0 * loss_lat)
                    val_loss_sum += float(val_step)
                    val_count += 1
            if val_count > 0:
                avg_val_loss = val_loss_sum / val_count

        scheduler.step()

        # Purge cuDNN workspace cache at epoch boundary to prevent fragmentation
        # over long training runs (the root cause of CUDNN_STATUS_INTERNAL_ERROR).
        if dev.type == "cuda":
            torch.cuda.empty_cache()

        epoch_stats = {
            "epoch": float(epoch),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_stats)

        epoch_pbar.set_postfix(
            train=f"{avg_train_loss:.3f}",
            val=f"{avg_val_loss:.3f}",
            best=f"{best_loss:.3f}",
        )

        checkpoint_data = {
            "epoch": epoch,
            "pipeline_state_dict": pipeline.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "prosody_state_dict": prosody_head.state_dict(),
            "fusion_state_dict": fusion.state_dict(),
            "cleanser_state_dict": cleanser.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "config": {
                "lr": lr,
                "batch_size": batch_size,
                "crop_sec": crop_sec,
            },
        }

        # Save latest
        torch.save(checkpoint_data, ckpt_dir / "stage2_adapter_latest.pt")

        # Save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(checkpoint_data, best_ckpt_path)

    with open(ckpt_dir / "stage2_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_ckpt_path


def main() -> None:
    """CLI entrypoint for Stage 2 training."""
    parser = argparse.ArgumentParser(
        description="Train Stage 2 Personalized Adapter on target persona."
    )
    parser.add_argument(
        "--target-manifest",
        type=str,
        default="data/hu_tao/processed/train_manifest.json",
        help="Path to target persona train manifest.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="data/hu_tao/processed/val_manifest.json",
        help="Path to target persona val manifest.",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=str,
        default="checkpoints/stage1_cleanser_best.pt",
        help="Path to pretrained Stage 1 cleanser checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--crop-sec", type=float, default=2.56)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    train_stage2(
        target_manifest=args.target_manifest,
        val_manifest=args.val_manifest,
        stage1_checkpoint=args.stage1_checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        crop_sec=args.crop_sec,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
