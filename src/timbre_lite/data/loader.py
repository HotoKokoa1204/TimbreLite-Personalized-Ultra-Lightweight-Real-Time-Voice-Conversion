"""Audio loading and saving utilities supporting multiple container formats."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf


def load_audio(
    path: str | Path,
    target_sr: int | None = None,
) -> tuple[torch.Tensor, int]:
    """Load audio file into a 1D float32 tensor scaled to [-1.0, 1.0].

    Supports common containers (e.g. .m4a, .mp3, .wav, .flac) via PyAV,
    with fallback to soundfile for standard uncompressed formats.

    Args:
        path: Path to the audio file.
        target_sr: Optional target sampling rate in Hz. If specified and
            different from the source sample rate, the audio will be resampled.

    Returns:
        Tuple of (audio_tensor, sample_rate) where audio_tensor has shape
        (num_samples,) and dtype torch.float32.

    Raises:
        FileNotFoundError: If the specified audio file does not exist.
        RuntimeError: If audio decoding fails.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Fast-path for .wav files using soundfile directly (50x faster)
    if file_path.suffix.lower() == ".wav":
        try:
            data, sr = sf.read(str(file_path), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=-1)
            tensor = torch.from_numpy(data).to(dtype=torch.float32)
            if target_sr is not None and target_sr != sr:
                tensor = taf.resample(tensor, orig_freq=sr, new_freq=target_sr)
                return tensor, target_sr
            return tensor, sr
        except Exception:
            pass

    # Primary decoder: PyAV for robust container decoding (m4a, aac, mp3)
    try:
        import av

        container = av.open(str(file_path))
        try:
            if not container.streams.audio:
                raise RuntimeError(f"No audio stream found in {file_path}")
            stream = container.streams.audio[0]
            native_sr: int = stream.rate or 48000

            resampler = av.AudioResampler(
                format="flt",
                layout="mono",
                rate=target_sr or native_sr,
            )

            chunks: list[np.ndarray] = []
            for frame in container.decode(audio=0):
                for r_frame in resampler.resample(frame):
                    chunks.append(r_frame.to_ndarray())
            for r_frame in resampler.resample(None):
                chunks.append(r_frame.to_ndarray())
        finally:
            container.close()

        if chunks:
            full_audio = np.concatenate(chunks, axis=1).squeeze(0)
            tensor = torch.from_numpy(full_audio).to(dtype=torch.float32)
            final_sr = target_sr or native_sr
            return tensor, final_sr
    except Exception:
        # Fallback to soundfile if PyAV fails or is unavailable
        pass

    # Soundfile fallback
    data, sr = sf.read(str(file_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=-1)
    tensor = torch.from_numpy(data).to(dtype=torch.float32)

    if target_sr is not None and target_sr != sr:
        tensor = taf.resample(tensor, orig_freq=sr, new_freq=target_sr)
        return tensor, target_sr

    return tensor, sr


def save_wav(
    path: str | Path,
    audio: torch.Tensor,
    sample_rate: int,
) -> None:
    """Save a 1D or 2D audio tensor to a 16-bit PCM WAV file.

    Args:
        path: Target file path to write.
        audio: 1D tensor of shape (num_samples,) or 2D (1, num_samples).
        sample_rate: Audio sampling rate in Hz.
    """
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if audio.ndim == 2:
        audio = audio.squeeze(0)
    arr = audio.detach().cpu().numpy().astype(np.float32)

    # Clamp audio to prevent wrap-around distortion
    arr = np.clip(arr, -1.0, 1.0)
    sf.write(str(target_path), arr, sample_rate, subtype="PCM_16")
