"""Unified Voice Conversion inference runner supporting batch and streaming."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from timbre_lite.codec.candidate import EnCodec24kCandidate
from timbre_lite.data.loader import load_audio, save_wav
from timbre_lite.modules.adapter import (
    DualStreamFusion,
    FullPersonalizedPipeline,
    InGraphProsodyHead,
    PersonalizedAdapter,
    PipelineStreamingState,
)
from timbre_lite.modules.cleanser import ContentCleanser


class VoiceConverter:
    """Unified voice conversion engine executing FullPersonalizedPipeline.

    Supports both full-utterance batch mode and causal chunk-by-chunk streaming mode
    with exact temporal state persistence.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Initialize VoiceConverter and optionally load persona weights.

        Args:
            checkpoint_path: Path to trained Stage 2 checkpoint (.pt).
            device: Target torch device (CUDA or CPU).
        """
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.codec = EnCodec24kCandidate()
        self.cleanser = ContentCleanser(in_dim=128, content_dim=64)
        self.prosody_head = InGraphProsodyHead(in_dim=128, prosody_dim=16)
        self.fusion = DualStreamFusion(content_dim=64, prosody_dim=16)
        self.adapter = PersonalizedAdapter(
            in_dim=64, out_dim=128, hidden_dim=64, tcn_layers=4, gru_hidden=64
        )

        self.pipeline = FullPersonalizedPipeline(
            codec=self.codec,
            cleanser=self.cleanser,
            prosody_head=self.prosody_head,
            fusion=self.fusion,
            adapter=self.adapter,
        ).to(self.device)

        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            self.load_checkpoint(checkpoint_path)

        self.pipeline.eval()

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load trained weights from a checkpoint dictionary.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "pipeline_state_dict" in ckpt:
            self.pipeline.load_state_dict(ckpt["pipeline_state_dict"])
        else:
            if "adapter_state_dict" in ckpt:
                self.adapter.load_state_dict(ckpt["adapter_state_dict"])
            if "cleanser_state_dict" in ckpt:
                self.cleanser.load_state_dict(ckpt["cleanser_state_dict"])
            if "prosody_state_dict" in ckpt:
                self.prosody_head.load_state_dict(ckpt["prosody_state_dict"])
            if "fusion_state_dict" in ckpt:
                self.fusion.load_state_dict(ckpt["fusion_state_dict"])

    def convert_utterance(self, audio_24k: torch.Tensor | np.ndarray) -> torch.Tensor:
        """Convert a complete 24kHz audio utterance to target voice.

        Args:
            audio_24k: 1D audio waveform at 24,000 Hz.

        Returns:
            Converted 1D audio tensor at 24,000 Hz on CPU.
        """
        if isinstance(audio_24k, np.ndarray):
            t_audio = torch.from_numpy(audio_24k).float()
        else:
            t_audio = audio_24k.float()

        orig_len = len(t_audio)
        if t_audio.ndim == 1:
            inp = t_audio.unsqueeze(0).unsqueeze(0).to(self.device)
        elif t_audio.ndim == 2:
            inp = t_audio.unsqueeze(1).to(self.device)
        else:
            inp = t_audio.to(self.device)

        codec_model: Any = self.pipeline.codec.model
        with torch.no_grad():
            z = codec_model.encoder(inp)
            z_adapted, _, _ = self.pipeline.forward_sequence(z)
            y_recon = codec_model.decoder(z_adapted).squeeze()

        if y_recon.ndim == 0:
            y_recon = y_recon.unsqueeze(0)
        res: torch.Tensor = y_recon[:orig_len].detach().cpu()

        # Match output energy to input energy
        in_rms = torch.sqrt(torch.mean(t_audio**2))
        out_rms = torch.sqrt(torch.mean(res**2))
        if in_rms > 1e-4 and out_rms > 1e-5:
            gain = in_rms / (out_rms + 1e-8)
            res = res * gain
            peak = torch.max(torch.abs(res))
            if peak > 0.99:
                res = res * (0.99 / peak)

        return res

    def convert_stream(
        self, audio_chunks: Iterable[torch.Tensor]
    ) -> Iterator[torch.Tensor]:
        """Convert a stream of 320-sample audio chunks sequentially.

        Args:
            audio_chunks: Iterable yielding 320-sample audio chunks.

        Yields:
            Converted 320-sample audio chunks.
        """
        state: PipelineStreamingState | None = None

        with torch.no_grad():
            for chunk in audio_chunks:
                if chunk.ndim == 1:
                    c_in = chunk.unsqueeze(0).unsqueeze(0).to(self.device)
                elif chunk.ndim == 2:
                    c_in = chunk.unsqueeze(1).to(self.device)
                else:
                    c_in = chunk.to(self.device)

                out_chunk, state = self.pipeline.step_audio_chunk(c_in, state)
                yield out_chunk.squeeze().detach().cpu()

    def convert_file(
        self,
        input_path: str | Path,
        output_path: str | Path,
        streaming: bool = False,
    ) -> Path:
        """Convert an audio file from disk to target voice and save as WAV.

        Args:
            input_path: Path to input audio file (WAV, MP3, M4A, etc.).
            output_path: Path to output 24kHz WAV file.
            streaming: Whether to process chunk-by-chunk in streaming mode.

        Returns:
            Path to the saved WAV file.
        """
        in_p = Path(input_path)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        audio_24k, sr = load_audio(in_p, target_sr=24000)
        orig_len = len(audio_24k)

        if streaming:
            num_chunks = (orig_len + 319) // 320
            chunks = []
            for i in range(num_chunks):
                c = audio_24k[i * 320 : (i + 1) * 320]
                if len(c) < 320:
                    c = torch.nn.functional.pad(c, (0, 320 - len(c)))
                chunks.append(c)

            out_chunks: list[torch.Tensor] = list(self.convert_stream(chunks))
            if out_chunks:
                converted = torch.cat(out_chunks, dim=-1)[:orig_len]
            else:
                converted = torch.zeros(0, dtype=torch.float32)
        else:
            converted = self.convert_utterance(audio_24k)

        # Match output energy with input audio
        in_rms = torch.sqrt(torch.mean(audio_24k**2))
        out_rms = torch.sqrt(torch.mean(converted**2))
        if in_rms > 1e-4 and out_rms > 1e-5:
            gain = in_rms / (out_rms + 1e-8)
            converted = converted * gain
            peak = torch.max(torch.abs(converted))
            if peak > 0.99:
                converted = converted * (0.99 / peak)

        save_wav(out_p, converted, sample_rate=24000)
        return out_p
