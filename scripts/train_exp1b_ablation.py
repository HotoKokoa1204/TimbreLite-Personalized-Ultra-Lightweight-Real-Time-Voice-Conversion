"""EXP-1B: Controlled 3-Way Ablation Training Script (Model A-ctrl, Model B, Model C)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.modules.adapter import (
    DualStreamFusion,
    ExplicitF0Encoder,
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


def get_file_hash(path: Path) -> str:
    """Compute SHA256 prefix hash of a file."""
    if not path.exists():
        return "none"
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def train_single_ablation_model(
    model_type: str,
    target_train_manifest: str | Path = "data/hu_tao/processed/train_manifest.json",
    target_val_manifest: str | Path = "data/hu_tao/processed/val_manifest.json",
    stage2_init_checkpoint: str | Path = "checkpoints/stage2_adapter_best.pt",
    f0_config_path: str | Path = "outputs/f0_production_config.json",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 2e-4,
    crop_sec: float = 2.56,
    seed: int = 42,
    output_base_dir: str | Path = "checkpoints",
    device: torch.device | str | None = None,
) -> Path:
    """Run strictly controlled training for one EXP-1B ablation variant.

    Args:
        model_type: 'model_a_control', 'model_b_f0', or 'model_c_gated'.
        target_train_manifest: Path to target train manifest.
        target_val_manifest: Path to target val manifest.
        stage2_init_checkpoint: Path to shared Stage 2 Epoch-82 checkpoint.
        f0_config_path: Path to frozen production F0 config.
        epochs: Training budget (strictly 50 epochs).
        batch_size: Batch size (strictly 8).
        lr: Learning rate (strictly 2e-4).
        crop_sec: Random crop duration in seconds (strictly 2.56s).
        seed: Random seed (strictly 42).
        output_base_dir: Directory to save experiment checkpoints.
        device: Target compute device.

    Returns:
        Path to best checkpoint file.
    """
    assert model_type in (
        "model_a_control",
        "model_b_f0",
        "model_c_gated",
    ), f"Invalid model_type {model_type}"

    # Set strict deterministic seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev = (
        torch.device(device)
        if device is not None
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.empty_cache()

    out_dir = Path(output_base_dir) / f"exp1b_{model_type}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(f"STARTING EXP-1B TRAINING: {model_type.upper()}")
    print(f"Device: {dev} | Seed: {seed} | Epochs: {epochs} | Batch: {batch_size}")
    print(f"Output Directory: {out_dir}")
    print("=" * 75)

    # 1. Initialize core submodules
    codec = EnCodec24kCandidate()
    cleanser = ContentCleanser(in_dim=128, content_dim=64)
    prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
    fusion = DualStreamFusion(content_dim=64, prosody_dim=16)

    # 2. Configure model-specific architecture
    use_f0 = model_type in ("model_b_f0", "model_c_gated")
    use_gated_skip = model_type == "model_c_gated"

    f0_encoder = (
        ExplicitF0Encoder(in_dim=3, hidden_dim=32, out_dim=64) if use_f0 else None
    )

    adapter = PersonalizedAdapter(
        in_dim=64,
        out_dim=128,
        hidden_dim=64,
        tcn_layers=4,
        gru_hidden=64,
        use_gated_skip=use_gated_skip,
        initial_skip_gate=-4.0,
    )

    pipeline = FullPersonalizedPipeline(
        codec=codec,
        cleanser=cleanser,
        prosody_head=prosody_head,
        fusion=fusion,
        adapter=adapter,
        f0_encoder=f0_encoder,
    ).to(dev)

    # 3. Load shared weights from Stage-2 Epoch-82 checkpoint
    init_ckpt_p = Path(stage2_init_checkpoint)
    assert init_ckpt_p.is_file(), f"Stage 2 init checkpoint {init_ckpt_p} not found"
    print(f"Loading base weights from: {init_ckpt_p}")
    init_ckpt = torch.load(init_ckpt_p, map_location="cpu", weights_only=False)

    if "adapter_state_dict" in init_ckpt:
        # Load adapter, ignoring skip_gate if not present in init ckpt
        missing, unexpected = adapter.load_state_dict(
            init_ckpt["adapter_state_dict"], strict=False
        )
        if use_gated_skip:
            assert "skip_gate" in missing, "Expected skip_gate to be missing in base ckpt"
            print("Successfully initialized learnable skip_gate to -4.0 (sigma ~= 0.018)")
    if "prosody_state_dict" in init_ckpt:
        prosody_head.load_state_dict(init_ckpt["prosody_state_dict"])
    if "fusion_state_dict" in init_ckpt:
        fusion.load_state_dict(init_ckpt["fusion_state_dict"])
    if "cleanser_state_dict" in init_ckpt:
        cleanser.load_state_dict(init_ckpt["cleanser_state_dict"])

    # Content Cleanser remains strictly frozen
    for param in cleanser.parameters():
        param.requires_grad = False
    cleanser.eval()

    trainable_params = pipeline.count_trainable_parameters()
    print(f"Model {model_type} initialized: {trainable_params:,} trainable parameters.")
    assert trainable_params < 250000, f"Exceeded 250K budget: {trainable_params}"

    # 4. Setup Optimizer & Cosine Scheduler
    trainer = AdapterTrainer(pipeline=pipeline, lr=lr, device=dev)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=epochs, eta_min=1e-5
    )

    # 5. Datasets & Loaders
    train_ds = VoiceConversionDataset(
        manifest_path=target_train_manifest,
        crop_sec=crop_sec,
        crop_strategy=CropStrategy.RANDOM,
        load_features=False,
        load_f0=use_f0,
        f0_dir="data/features/f0/hu_tao",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_voice_batch,
    )

    val_ds = VoiceConversionDataset(
        manifest_path=target_val_manifest,
        crop_sec=crop_sec,
        crop_strategy=CropStrategy.CENTER,
        load_features=False,
        load_f0=use_f0,
        f0_dir="data/features/f0/hu_tao",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_voice_batch,
    )

    # Lineage Hashes
    init_ckpt_hash = get_file_hash(init_ckpt_p)
    train_m_hash = get_file_hash(Path(target_train_manifest))
    val_m_hash = get_file_hash(Path(target_val_manifest))
    f0_cfg_hash = get_file_hash(Path(f0_config_path))

    best_val_loss = float("inf")
    best_ckpt_path = out_dir / "best_adapter.pt"
    history: list[dict[str, Any]] = []

    epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Train {model_type}")
    for epoch in epoch_pbar:
        # Train epoch
        pipeline.train()
        train_loss_sum = 0.0
        train_steps = 0
        for b_idx, batch in enumerate(train_loader):
            audio_24k = batch["audio_24k"]
            f0_3d = batch.get("f0_3d")
            loss_dict = trainer.train_step(
                target_audio_24k=audio_24k,
                mask=None,
                f0_3d=f0_3d,
            )
            step_loss = loss_dict.get("loss_adapter_total", loss_dict.get("total_loss", 0.0))
            if step_loss > 0:
                train_loss_sum += step_loss
                train_steps += 1
            if (b_idx + 1) % 10 == 0 or (b_idx + 1) == len(train_loader):
                print(
                    f"[{model_type}] Epoch {epoch}/{epochs} | Step {b_idx + 1}/{len(train_loader)} | Train Loss: {step_loss:.4f}",
                    flush=True,
                )

        avg_train_loss = train_loss_sum / max(train_steps, 1)

        # Validation epoch
        pipeline.eval()
        val_loss_sum = 0.0
        val_steps = 0
        codec_model: Any = pipeline.codec.model
        with torch.no_grad():
            for v_batch in val_loader:
                audio = v_batch["audio_24k"].to(dev)
                audio_in = audio.unsqueeze(1) if audio.ndim == 2 else audio
                f0_3d_val = v_batch["f0_3d"].to(dev) if use_f0 and "f0_3d" in v_batch else None

                z_target = codec_model.encoder(audio_in)
                z_adapted, _, _ = pipeline.forward_sequence(z_target, f0_seq=f0_3d_val)
                y_recon = codec_model.decoder(z_adapted).squeeze(1)

                loss_stft = trainer.stft_loss(y_recon, audio)
                loss_lat = trainer.latent_loss(z_adapted, z_target)
                val_step = float(loss_stft + (2.0 * loss_lat))
                val_loss_sum += val_step
                val_steps += 1

        avg_val_loss = val_loss_sum / max(val_steps, 1)
        scheduler.step()

        if dev.type == "cuda":
            torch.cuda.empty_cache()

        gate_val: float | None = None
        if use_gated_skip and adapter.skip_gate is not None:
            gate_val = float(torch.sigmoid(adapter.skip_gate).item())

        stat = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": float(scheduler.get_last_lr()[0]),
            "skip_gate_sigma": gate_val,
        }
        history.append(stat)

        postfix: dict[str, str] = {
            "train": f"{avg_train_loss:.3f}",
            "val": f"{avg_val_loss:.3f}",
            "best": f"{best_val_loss:.3f}",
        }
        if gate_val is not None:
            postfix["gate"] = f"{gate_val:.4f}"
        epoch_pbar.set_postfix(**postfix)

        # Save checkpoint payload with full lineage metadata
        ckpt_data: dict[str, Any] = {
            "epoch": epoch,
            "model_type": model_type,
            "pipeline_state_dict": pipeline.state_dict(),
            "adapter_state_dict": adapter.state_dict(),
            "prosody_state_dict": prosody_head.state_dict(),
            "fusion_state_dict": fusion.state_dict(),
            "cleanser_state_dict": cleanser.state_dict(),
            "f0_encoder_state_dict": f0_encoder.state_dict() if f0_encoder is not None else None,
            "optimizer_state_dict": trainer.optimizer.state_dict(),
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "skip_gate_sigma": gate_val,
            "lineage": {
                "initial_checkpoint": str(init_ckpt_p),
                "initial_checkpoint_hash": init_ckpt_hash,
                "train_manifest_hash": train_m_hash,
                "val_manifest_hash": val_m_hash,
                "f0_config_hash": f0_cfg_hash,
                "seed": seed,
                "batch_size": batch_size,
                "crop_sec": crop_sec,
                "lr": lr,
            },
        }

        torch.save(ckpt_data, out_dir / "latest_adapter.pt")

        # Checkpoint selection: best_val_loss over 50 epochs
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(ckpt_data, best_ckpt_path)

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Training completed for {model_type}. Best Val Loss: {best_val_loss:.4f}")
    return best_ckpt_path


def main() -> None:
    """CLI driver for EXP-1B ablation training."""
    parser = argparse.ArgumentParser(description="EXP-1B Controlled Ablation Trainer")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all", "model_a_control", "model_b_f0", "model_c_gated"],
        help="Which ablation model to train (default: 'all')",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default 50)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    args = parser.parse_args()

    models_to_train = (
        ["model_a_control", "model_b_f0", "model_c_gated"]
        if args.model == "all"
        else [args.model]
    )

    for m in models_to_train:
        train_single_ablation_model(
            model_type=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
        )


if __name__ == "__main__":
    main()
