"""Unit tests for VoiceConverter inference engine and CLI runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from timbre_lite.data.loader import save_wav
from timbre_lite.inference.runner import VoiceConverter


def test_voice_converter_init_and_convert_utterance() -> None:
    """Verify VoiceConverter initializes on CPU and converts 24kHz audio."""
    converter = VoiceConverter(device="cpu")
    sr = 24000
    duration_sec = 0.5
    num_samples = int(sr * duration_sec)
    audio = 0.1 * torch.sin(
        2 * 3.14159 * 300 * torch.linspace(0, duration_sec, num_samples)
    )

    converted = converter.convert_utterance(audio)
    assert isinstance(converted, torch.Tensor)
    assert not torch.isnan(converted).any()
    # EnCodec downsamples 320x and upsamples 320x, so length matches
    assert len(converted) == num_samples


def test_voice_converter_streaming_chunks() -> None:
    """Verify chunk-by-chunk streaming conversion produces valid 320-sample slices."""
    converter = VoiceConverter(device="cpu")
    num_chunks = 5
    chunks = [torch.randn(320) * 0.05 for _ in range(num_chunks)]

    stream_out = list(converter.convert_stream(chunks))
    assert len(stream_out) == num_chunks
    for out_chunk in stream_out:
        assert isinstance(out_chunk, torch.Tensor)
        assert len(out_chunk) == 320
        assert not torch.isnan(out_chunk).any()


def test_voice_converter_file_conversion_roundtrip() -> None:
    """Verify audio file conversion to disk in both batch and streaming modes."""
    converter = VoiceConverter(device="cpu")
    sr = 24000
    audio = torch.randn(sr) * 0.05

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_p = Path(tmp_dir)
        in_wav = tmp_p / "input_speech.wav"
        out_batch = tmp_p / "out_batch.wav"
        out_stream = tmp_p / "out_stream.wav"

        save_wav(in_wav, audio, sr)
        assert in_wav.is_file()

        converter.convert_file(in_wav, out_batch, streaming=False)
        assert out_batch.is_file()

        converter.convert_file(in_wav, out_stream, streaming=True)
        assert out_stream.is_file()


def test_voice_converter_checkpoint_loading_roundtrip() -> None:
    """Verify loading custom weights into VoiceConverter pipeline."""
    converter_source = VoiceConverter(device="cpu")
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_stage2_ckpt.pt"
        torch.save(
            {
                "pipeline_state_dict": converter_source.pipeline.state_dict(),
                "epoch": 1,
            },
            ckpt_path,
        )

        converter_dest = VoiceConverter(checkpoint_path=ckpt_path, device="cpu")
        assert converter_dest.pipeline is not None
