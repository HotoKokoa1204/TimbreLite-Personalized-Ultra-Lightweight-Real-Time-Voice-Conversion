"""Stage 1: Offline ContentVec phonetic distillation training script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.distillation.loss import DistillationProjectionHead
from timbre_lite.modules.cleanser import ContentCleanser
from timbre_lite.training.dataset import (
    CropStrategy,
    VoiceConversionDataset,
    collate_voice_batch,
)
from timbre_lite.training.trainer import DistillationTrainer


def train_stage1(
    train_manifest: str | Path,
    val_manifest: str | Path | None = None,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 3e-4,
    crop_sec: float = 2.56,
    checkpoint_dir: str | Path = "checkpoints",
    device: torch.device | str | None = None,
) -> Path:
    """Run Stage 1 distillation training loop and save best checkpoint.

    Args:
        train_manifest: Path to train manifest with cached features.
        val_manifest: Optional path to val manifest.
        epochs: Number of training epochs (default: 100).
        batch_size: DataLoader mini-batch size.
        lr: Learning rate for AdamW optimizer.
        crop_sec: Training crop duration in seconds.
        checkpoint_dir: Directory to save checkpoint files.
        device: Target compute device.

    Returns:
        Path to best saved checkpoint file.
    """
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"Initializing Stage 1 training ({epochs} epochs) on device: {dev}")
    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    proj_head = DistillationProjectionHead(in_dim=64, out_dim=768)

    trainer = DistillationTrainer(
        cleanser=cleanser,
        proj_head=proj_head,
        codec=codec,
        lr=lr,
        device=dev,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs, eta_min=1e-5
    )

    train_ds = VoiceConversionDataset(
        manifest_path=train_manifest,
        crop_sec=crop_sec,
        crop_strategy=CropStrategy.RANDOM,
        load_features=True,
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
            load_features=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_voice_batch,
        )

    best_loss = float("inf")
    best_ckpt_path = ckpt_dir / "stage1_cleanser_best.pt"
    history: list[dict[str, float]] = []

    epoch_pbar = tqdm(
        range(1, epochs + 1),
        desc="Stage 1 Progress (100 Epochs)",
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
            if "teacher_features" not in batch:
                continue

            metrics = trainer.train_step(
                audio_24k=batch["audio_24k"],
                teacher_features=batch["teacher_features"],
                mask=batch.get("mask"),
            )
            step_loss = metrics.get("loss_distill_total", 0.0)
            total_loss += step_loss
            num_batches += 1
            cos_loss = metrics.get("loss_distill_cos", 1.0)
            cos_sim = max(0.0, 1.0 - cos_loss)
            pbar.set_postfix(
                {
                    "loss": f"{step_loss:.4f}",
                    "sim": f"{cos_sim:.4f}",
                }
            )

        avg_train_loss = total_loss / max(num_batches, 1)

        # Validation loop
        avg_val_loss = avg_train_loss
        avg_val_sim = 0.0
        if val_loader is not None:
            val_loss_sum = 0.0
            val_cos_sum = 0.0
            val_count = 0
            cleanser.eval()
            proj_head.eval()
            with torch.no_grad():
                for v_batch in val_loader:
                    if "teacher_features" not in v_batch:
                        continue
                    audio = v_batch["audio_24k"].to(dev)
                    t_feat = v_batch["teacher_features"].to(dev)
                    mask = v_batch.get("mask")
                    if mask is not None:
                        mask = mask.to(dev)

                    if audio.ndim == 2:
                        audio = audio.unsqueeze(1)
                    z = codec.model.encoder(audio)
                    c = cleanser.forward_sequence(z)
                    p = proj_head(c)
                    loss, val_m = trainer.loss_fn(p, t_feat, mask=mask)
                    val_loss_sum += float(loss)
                    val_cos_sum += float(val_m.get("loss_distill_cos", 1.0))
                    val_count += 1
            if val_count > 0:
                avg_val_loss = val_loss_sum / val_count
                avg_val_sim = max(0.0, 1.0 - (val_cos_sum / val_count))

        scheduler.step()

        epoch_stats = {
            "epoch": float(epoch),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_sim": avg_val_sim,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(epoch_stats)

        epoch_pbar.set_postfix(
            train=f"{avg_train_loss:.4f}",
            val=f"{avg_val_loss:.4f}",
            sim=f"{avg_val_sim:.4f}",
            best=f"{best_loss:.4f}",
        )

        checkpoint_data = {
            "epoch": epoch,
            "cleanser_state_dict": cleanser.state_dict(),
            "proj_head_state_dict": proj_head.state_dict(),
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
        torch.save(checkpoint_data, ckpt_dir / "stage1_cleanser_latest.pt")

        # Save best
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(checkpoint_data, best_ckpt_path)

    # Write training history log
    with open(ckpt_dir / "stage1_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_ckpt_path


def main() -> None:
    """CLI entrypoint for Stage 1 training."""
    parser = argparse.ArgumentParser(
        description=(
            "Train Stage 1 Content Cleanser with offline ContentVec distillation."
        )
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="data/features/my_voice/train_manifest_with_features.json",
        help="Path to training manifest.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="data/features/my_voice/val_manifest_with_features.json",
        help="Path to validation manifest.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--crop-sec", type=float, default=2.56)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    train_stage1(
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        crop_sec=args.crop_sec,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
