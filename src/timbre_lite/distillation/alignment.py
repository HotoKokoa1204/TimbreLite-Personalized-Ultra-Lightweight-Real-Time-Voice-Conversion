"""1D Temporal linear interpolation aligning 50Hz ContentVec to 75Hz EnCodec."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def align_teacher_features(
    teacher_features: torch.Tensor,
    target_frames: int,
) -> torch.Tensor:
    """Resample 50Hz teacher representation to match 75Hz EnCodec frames.

    Applies 1D linear interpolation to map from ContentVec temporal frame rate
    (50 Hz) to the exact number of frames in the EnCodec continuous latent space
    (75 Hz, target_frames = num_samples_24k // 320).

    Args:
        teacher_features: Feature tensor of shape (batch, channels, time_in) or
            (channels, time_in) where channels is typically 768.
        target_frames: Desired output frame count along time dimension (75 Hz).

    Returns:
        Interpolated feature tensor of shape (batch, channels, target_frames)
        matching target_frames with float32 precision.

    Raises:
        ValueError: If teacher_features has fewer than 2 dimensions or if
            target_frames <= 0.
    """
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    was_2d = False
    if teacher_features.ndim == 2:
        # (channels, time) -> (1, channels, time)
        was_2d = True
        teacher_features = teacher_features.unsqueeze(0)
    elif teacher_features.ndim != 3:
        raise ValueError(
            f"Expected 2D or 3D tensor, got shape {tuple(teacher_features.shape)}"
        )

    # In 3D: (batch, channels, time_in)
    time_in = teacher_features.shape[-1]
    if time_in == target_frames:
        aligned = teacher_features
    else:
        aligned = F.interpolate(
            teacher_features,
            size=target_frames,
            mode="linear",
            align_corners=False,
        )

    if was_2d:
        return aligned.squeeze(0)
    return aligned
