"""Stateful streaming execution engines for EnCodec causal encoder and decoder."""

from __future__ import annotations

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from timbre_lite.codec.state import CodecState


class StatefulSEANetEncoder:
    """Stateful streaming wrapper for Meta SEANet causal encoder.

    Executes frame-by-frame (320 samples -> 1 latent frame) in O(1) time
    with exact mathematical equivalence to batch full-utterance forward pass.
    """

    def __init__(self, encoder: nn.Module) -> None:
        """Initialize stateful encoder wrapper.

        Args:
            encoder: Pretrained SEANetEncoder module from EncodecModel.
        """
        self.encoder = encoder

    def init_state(
        self,
        state: CodecState,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> None:
        """Initialize encoder FIFO conv buffers and LSTM hidden states.

        Args:
            state: Target CodecState container.
            batch_size: Number of parallel audio streams.
            device: Target torch device for buffers.
        """
        state.encoder_conv_states["0"] = torch.zeros(batch_size, 1, 6, device=device)
        state.encoder_conv_states["1.block.1"] = torch.zeros(
            batch_size, 32, 2, device=device
        )
        state.encoder_conv_states["3"] = torch.zeros(batch_size, 32, 2, device=device)
        state.encoder_conv_states["4.block.1"] = torch.zeros(
            batch_size, 64, 2, device=device
        )
        state.encoder_conv_states["6"] = torch.zeros(batch_size, 64, 4, device=device)
        state.encoder_conv_states["7.block.1"] = torch.zeros(
            batch_size, 128, 2, device=device
        )
        state.encoder_conv_states["9"] = torch.zeros(batch_size, 128, 5, device=device)
        state.encoder_conv_states["10.block.1"] = torch.zeros(
            batch_size, 256, 2, device=device
        )
        state.encoder_conv_states["12"] = torch.zeros(batch_size, 256, 8, device=device)
        h = torch.zeros(2, batch_size, 512, device=device)
        c = torch.zeros(2, batch_size, 512, device=device)
        state.encoder_lstm_states["13"] = (h, c)
        state.encoder_conv_states["15"] = torch.zeros(batch_size, 512, 6, device=device)

    def step(
        self, audio_chunk: torch.Tensor, state: CodecState
    ) -> tuple[torch.Tensor, CodecState]:
        """Encode a single 320-sample audio chunk into a 128-d latent frame.

        Args:
            audio_chunk: Raw audio tensor of shape (batch, 1, 320).
            state: Streaming state holding FIFO conv buffers and LSTM states.

        Returns:
            Tuple of (latent_chunk, next_state) where latent_chunk has shape
            (batch, 128, 1).
        """
        # Ensure state buffers are initialized for batch_size and device
        batch_size = audio_chunk.shape[0]
        device = audio_chunk.device
        if (
            "0" not in state.encoder_conv_states
            or state.encoder_conv_states["0"].shape[0] != batch_size
        ):
            self.init_state(state, batch_size=batch_size, device=device)

        m: tp.Any = self.encoder.model

        # Layer 0: SConv1d (k=7, s=1, pad=6)
        buf0 = state.encoder_conv_states["0"]
        x0 = torch.cat([buf0, audio_chunk], dim=-1)
        state.encoder_conv_states["0"] = x0[:, :, -6:]
        x = m[0].conv(x0)

        # Layer 1: SEANetResnetBlock
        res1 = x
        y1 = m[1].block[0](x)
        buf1 = state.encoder_conv_states["1.block.1"]
        y1_cat = torch.cat([buf1, y1], dim=-1)
        state.encoder_conv_states["1.block.1"] = y1_cat[:, :, -2:]
        y1 = m[1].block[1].conv(y1_cat)
        y1 = m[1].block[2](y1)
        y1 = m[1].block[3].conv(y1)
        x = m[1].shortcut(res1) + y1

        # Layer 2: ELU
        x = m[2](x)

        # Layer 3: SConv1d (k=4, s=2, pad=2)
        buf3 = state.encoder_conv_states["3"]
        x3 = torch.cat([buf3, x], dim=-1)
        state.encoder_conv_states["3"] = x3[:, :, -2:]
        x = m[3].conv(x3)

        # Layer 4: SEANetResnetBlock
        res4 = x
        y4 = m[4].block[0](x)
        buf4 = state.encoder_conv_states["4.block.1"]
        y4_cat = torch.cat([buf4, y4], dim=-1)
        state.encoder_conv_states["4.block.1"] = y4_cat[:, :, -2:]
        y4 = m[4].block[1].conv(y4_cat)
        y4 = m[4].block[2](y4)
        y4 = m[4].block[3].conv(y4)
        x = m[4].shortcut(res4) + y4

        # Layer 5: ELU
        x = m[5](x)

        # Layer 6: SConv1d (k=8, s=4, pad=4)
        buf6 = state.encoder_conv_states["6"]
        x6 = torch.cat([buf6, x], dim=-1)
        state.encoder_conv_states["6"] = x6[:, :, -4:]
        x = m[6].conv(x6)

        # Layer 7: SEANetResnetBlock
        res7 = x
        y7 = m[7].block[0](x)
        buf7 = state.encoder_conv_states["7.block.1"]
        y7_cat = torch.cat([buf7, y7], dim=-1)
        state.encoder_conv_states["7.block.1"] = y7_cat[:, :, -2:]
        y7 = m[7].block[1].conv(y7_cat)
        y7 = m[7].block[2](y7)
        y7 = m[7].block[3].conv(y7)
        x = m[7].shortcut(res7) + y7

        # Layer 8: ELU
        x = m[8](x)

        # Layer 9: SConv1d (k=10, s=5, pad=5)
        buf9 = state.encoder_conv_states["9"]
        x9 = torch.cat([buf9, x], dim=-1)
        state.encoder_conv_states["9"] = x9[:, :, -5:]
        x = m[9].conv(x9)

        # Layer 10: SEANetResnetBlock
        res10 = x
        y10 = m[10].block[0](x)
        buf10 = state.encoder_conv_states["10.block.1"]
        y10_cat = torch.cat([buf10, y10], dim=-1)
        state.encoder_conv_states["10.block.1"] = y10_cat[:, :, -2:]
        y10 = m[10].block[1].conv(y10_cat)
        y10 = m[10].block[2](y10)
        y10 = m[10].block[3].conv(y10)
        x = m[10].shortcut(res10) + y10

        # Layer 11: ELU
        x = m[11](x)

        # Layer 12: SConv1d (k=16, s=8, pad=8)
        buf12 = state.encoder_conv_states["12"]
        x12 = torch.cat([buf12, x], dim=-1)
        state.encoder_conv_states["12"] = x12[:, :, -8:]
        x = m[12].conv(x12)

        # Layer 13: SLSTM (2 layers, 512)
        x_perm = x.permute(2, 0, 1)
        lstm_out, next_lstm = m[13].lstm(x_perm, state.encoder_lstm_states["13"])
        state.encoder_lstm_states["13"] = next_lstm
        if m[13].skip:
            lstm_out = lstm_out + x_perm
        x = lstm_out.permute(1, 2, 0)

        # Layer 14: ELU
        x = m[14](x)

        # Layer 15: SConv1d (k=7, s=1, pad=6)
        buf15 = state.encoder_conv_states["15"]
        x15 = torch.cat([buf15, x], dim=-1)
        state.encoder_conv_states["15"] = x15[:, :, -6:]
        latent_chunk = tp.cast(torch.Tensor, m[15].conv(x15))

        return latent_chunk, state


class StatefulSEANetDecoder:
    """Stateful streaming wrapper for Meta SEANet causal decoder.

    Executes frame-by-frame (1 latent frame -> 320 samples) with exact causal
    overlap-add transposed convolutions and stateful recurrent LSTM.
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Initialize stateful decoder wrapper.

        Args:
            decoder: Pretrained SEANetDecoder module from EncodecModel.
        """
        self.decoder = decoder

    def init_state(
        self,
        state: CodecState,
        batch_size: int = 1,
        device: torch.device | None = None,
    ) -> None:
        """Initialize decoder FIFO conv buffers, overlap buffers, and LSTM state.

        Args:
            state: Target CodecState container.
            batch_size: Number of parallel audio streams.
            device: Target torch device for buffers.
        """
        state.decoder_conv_states["0"] = torch.zeros(batch_size, 128, 6, device=device)
        h = torch.zeros(2, batch_size, 512, device=device)
        c = torch.zeros(2, batch_size, 512, device=device)
        state.decoder_lstm_states["1"] = (h, c)
        state.decoder_overlap_states["3"] = torch.zeros(
            batch_size, 256, 8, device=device
        )
        state.decoder_conv_states["4.block.1"] = torch.zeros(
            batch_size, 256, 2, device=device
        )
        state.decoder_overlap_states["6"] = torch.zeros(
            batch_size, 128, 5, device=device
        )
        state.decoder_conv_states["7.block.1"] = torch.zeros(
            batch_size, 128, 2, device=device
        )
        state.decoder_overlap_states["9"] = torch.zeros(
            batch_size, 64, 4, device=device
        )
        state.decoder_conv_states["10.block.1"] = torch.zeros(
            batch_size, 64, 2, device=device
        )
        state.decoder_overlap_states["12"] = torch.zeros(
            batch_size, 32, 2, device=device
        )
        state.decoder_conv_states["13.block.1"] = torch.zeros(
            batch_size, 32, 2, device=device
        )
        state.decoder_conv_states["15"] = torch.zeros(batch_size, 32, 6, device=device)

    def _convtr_step(
        self,
        x: torch.Tensor,
        layer_key: str,
        sconvtr: nn.Module,
        state: CodecState,
    ) -> torch.Tensor:
        """Execute causal overlap-add step for transposed convolution.

        Args:
            x: Input feature chunk.
            layer_key: Unique identifier in state.decoder_overlap_states.
            sconvtr: EnCodec SConvTranspose1d layer.
            state: CodecState holding overlap buffer.

        Returns:
            Overlap-added output tensor of length x.shape[-1] * stride.
        """
        sconvtr_any: tp.Any = sconvtr
        convtr: tp.Any = sconvtr_any.convtr.convtr
        weight: torch.Tensor = convtr.weight
        bias: torch.Tensor | None = convtr.bias
        stride: int = convtr.stride[0]
        kernel_size: int = convtr.kernel_size[0]
        overlap_len = kernel_size - stride

        # Raw conv_transpose without bias
        raw: torch.Tensor = F.conv_transpose1d(x, weight, bias=None, stride=stride)
        out_len = x.shape[-1] * stride

        curr = raw[:, :, :out_len].clone()
        prev_overlap = state.decoder_overlap_states[layer_key]
        curr[:, :, :overlap_len] += prev_overlap
        if bias is not None:
            curr = curr + bias.view(1, -1, 1)

        next_overlap = raw[:, :, out_len : out_len + overlap_len]
        state.decoder_overlap_states[layer_key] = next_overlap
        return curr

    def step(
        self, latent_chunk: torch.Tensor, state: CodecState
    ) -> tuple[torch.Tensor, CodecState]:
        """Decode a single 128-d latent frame into 320 audio samples.

        Args:
            latent_chunk: Continuous latent of shape (batch, 128, 1).
            state: Streaming state holding FIFO conv buffers, overlap, and LSTM.

        Returns:
            Tuple of (audio_chunk, next_state) where audio_chunk has shape
            (batch, 1, 320).
        """
        batch_size = latent_chunk.shape[0]
        device = latent_chunk.device
        if (
            "0" not in state.decoder_conv_states
            or state.decoder_conv_states["0"].shape[0] != batch_size
        ):
            self.init_state(state, batch_size=batch_size, device=device)

        m: tp.Any = self.decoder.model

        # Layer 0: SConv1d (k=7, s=1, pad=6)
        buf0 = state.decoder_conv_states["0"]
        z0 = torch.cat([buf0, latent_chunk], dim=-1)
        state.decoder_conv_states["0"] = z0[:, :, -6:]
        x = m[0].conv(z0)

        # Layer 1: SLSTM (2 layers, 512)
        x_perm = x.permute(2, 0, 1)
        lstm_out, next_lstm = m[1].lstm(x_perm, state.decoder_lstm_states["1"])
        state.decoder_lstm_states["1"] = next_lstm
        if m[1].skip:
            lstm_out = lstm_out + x_perm
        x = lstm_out.permute(1, 2, 0)

        # Layer 2: ELU
        x = m[2](x)

        # Layer 3: SConvTranspose1d (stride=8, k=16)
        x = self._convtr_step(x, "3", m[3], state)

        # Layer 4: SEANetResnetBlock
        res4 = x
        y4 = m[4].block[0](x)
        buf4 = state.decoder_conv_states["4.block.1"]
        y4_cat = torch.cat([buf4, y4], dim=-1)
        state.decoder_conv_states["4.block.1"] = y4_cat[:, :, -2:]
        y4 = m[4].block[1].conv(y4_cat)
        y4 = m[4].block[2](y4)
        y4 = m[4].block[3].conv(y4)
        x = m[4].shortcut(res4) + y4

        # Layer 5: ELU
        x = m[5](x)

        # Layer 6: SConvTranspose1d (stride=5, k=10)
        x = self._convtr_step(x, "6", m[6], state)

        # Layer 7: SEANetResnetBlock
        res7 = x
        y7 = m[7].block[0](x)
        buf7 = state.decoder_conv_states["7.block.1"]
        y7_cat = torch.cat([buf7, y7], dim=-1)
        state.decoder_conv_states["7.block.1"] = y7_cat[:, :, -2:]
        y7 = m[7].block[1].conv(y7_cat)
        y7 = m[7].block[2](y7)
        y7 = m[7].block[3].conv(y7)
        x = m[7].shortcut(res7) + y7

        # Layer 8: ELU
        x = m[8](x)

        # Layer 9: SConvTranspose1d (stride=4, k=8)
        x = self._convtr_step(x, "9", m[9], state)

        # Layer 10: SEANetResnetBlock
        res10 = x
        y10 = m[10].block[0](x)
        buf10 = state.decoder_conv_states["10.block.1"]
        y10_cat = torch.cat([buf10, y10], dim=-1)
        state.decoder_conv_states["10.block.1"] = y10_cat[:, :, -2:]
        y10 = m[10].block[1].conv(y10_cat)
        y10 = m[10].block[2](y10)
        y10 = m[10].block[3].conv(y10)
        x = m[10].shortcut(res10) + y10

        # Layer 11: ELU
        x = m[11](x)

        # Layer 12: SConvTranspose1d (stride=2, k=4)
        x = self._convtr_step(x, "12", m[12], state)

        # Layer 13: SEANetResnetBlock
        res13 = x
        y13 = m[13].block[0](x)
        buf13 = state.decoder_conv_states["13.block.1"]
        y13_cat = torch.cat([buf13, y13], dim=-1)
        state.decoder_conv_states["13.block.1"] = y13_cat[:, :, -2:]
        y13 = m[13].block[1].conv(y13_cat)
        y13 = m[13].block[2](y13)
        y13 = m[13].block[3].conv(y13)
        x = m[13].shortcut(res13) + y13

        # Layer 14: ELU
        x = m[14](x)

        # Layer 15: SConv1d (k=7, s=1, pad=6)
        buf15 = state.decoder_conv_states["15"]
        x15 = torch.cat([buf15, x], dim=-1)
        state.decoder_conv_states["15"] = x15[:, :, -6:]
        audio_chunk = tp.cast(torch.Tensor, m[15].conv(x15))

        return audio_chunk, state
