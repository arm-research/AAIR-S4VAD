# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import math
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import median_filter
from torchvision.transforms.v2 import Compose
from s4vad.common.registries import register
from s4vad.vad.models.vad import VADAdapter
from s4vad.vad.utils import causal_median_filter, OnlineNormalizer
from utils import get_device
from einops import repeat




class S4D(nn.Module):
    """
    Generate convolution kernel from diagonal SSM parameters
    exp(-0.5*dt*T)<0.01 -> 1% of history after T frames
    dt_min = -ln(0.01) / (0.5T)
    """

    def __init__(self, d_model, T=21, N=64, dt_min=0.66, dt_max=1.0):
        super().__init__()
        self.device = get_device()
        self.T = T
        N2 = N//2
        self.register_buffer("const2", torch.tensor([2.0], device=self.device), persistent=False)
        dt = (dt_max / dt_min)**(torch.rand(d_model, device=self.device)) * dt_min
        A_real = -0.5 * torch.ones(d_model, N2, device=self.device)
        A_imag = math.pi * repeat(torch.arange(N2, device=self.device), 'n -> h n', h=d_model)
        A = A_real + 1j * A_imag
        self.register_buffer("dt", dt, persistent=False)
        self.register_buffer("A", A, persistent=False)
        self.B = 1
        self.C = nn.Parameter(torch.randn(d_model, N2, 2, dtype=torch.float, device=self.device))
        dA, dB = self.setup_step(self.dt, self.A, self.B)
        self.register_buffer("dA", dA)
        self.register_buffer("dB", dB)

    def setup_step(self, dt, A, B):
        """
        Precompute real-valued discrete parameters for stepping
        """
        dt_ = dt.unsqueeze(-1)                 # [d_model, 1]
        dA = torch.exp(A*dt_)                  # [d_model, N2] complex
        dB = (dA - 1)/A*B                      # [d_model, N2] complex
        # View as real
        dA = torch.view_as_real(dA)            # [d_model, N2, 2] real
        dB = torch.view_as_real(dB)            # [d_model, N2, 2] real
        return dA, dB

    def build_kernel(self, dt, A, B, C) -> torch.Tensor:
        # Materialize parameters
        dC = torch.view_as_complex(C)                                       # [d_model, N2] complex
        dt_ = dt.unsqueeze(-1)
        dtA = A * dt_                                                       # [d_model, N2] complex
        dB = (torch.exp(dtA) - 1) / A * B                                   # [d_model, N2] complex
        v = dtA.unsqueeze(-1) * torch.arange(self.T, device=dtA.device)     # [d_model, N2, T] complex
        # Vandermonde multiplication
        K = 2 * torch.einsum('dn, dnt -> dt', dC*dB, torch.exp(v)).real     # [d_model, T] real
        #K = 2 * ((dC * dB).unsqueeze(-1) * torch.exp(v)).sum(dim=1).real   # [d_model, T] real
        # We flip kernel because we implement real convolution by conv1d (which is more like correlation)
        K = K.flip(-1).unsqueeze(1)                                         # [d_model, 1, T]
        return K

    def single_step_matmul(self, dA, dB, C, u_t, state) -> torch.Tensor:
        """" Implementation using matmul """
        # Build a 2x2 real matrix that represents complex multiply by dA:
        # (a+ib)(x+iy) => [a -b; b a] @ [x; y]
        # M: [B, D, N, 2, 2]
        M = torch.stack([
                torch.stack([dA[..., 0], -dA[..., 1]], dim=-1),
                torch.stack([dA[..., 1], dA[..., 0]], dim=-1),],
                dim=-2)

        # state_vec: [B, D, N, 2, 1]
        state_vec = state.unsqueeze(-1)

        # z_vec: [B, D, N, 2, 1]
        z_vec = torch.matmul(M, state_vec)
        z = z_vec.squeeze(-1)  # [B, D, N, 2]

        u = u_t.unsqueeze(-1)
        dB_re = dB[..., 0].unsqueeze(-1)
        dB_im = dB[..., 1].unsqueeze(-1)

        # u_re/im: [B, D, N, 1] via bmm/matmul (still matmul op)
        u_re = torch.matmul(dB_re, u.unsqueeze(-1)).squeeze(-1)  # (N,1)@(1,1) -> (N,1)
        u_im = torch.matmul(dB_im, u.unsqueeze(-1)).squeeze(-1)

        # Update state (adds are usually counted only if tool counts elementwise; matmul already counted)
        next_state_re = z[..., 0] + u_re
        next_state_im = z[..., 1] + u_im

        # Output y = 2 * Re( sum_n C_n * next_state )
        # Express sums as matmul: (1xN)@(Nx1)
        # Make row vectors: [B, D, 1, N]
        C_re_row = C[..., 0].unsqueeze(-2)
        C_im_row = C[..., 1].unsqueeze(-2)

        # Make col vectors: [B, D, N, 1]
        ns_re_col = next_state_re.unsqueeze(-1)
        ns_im_col = next_state_im.unsqueeze(-1)

        # Each becomes [B, D, 1, 1]
        dot_re = torch.matmul(C_re_row, ns_re_col)
        dot_im = torch.matmul(C_im_row, ns_im_col)

        y_t = self.const2 * (dot_re - dot_im).squeeze(-1).squeeze(-1)  # [B, D]

        next_state = torch.stack([next_state_re, next_state_im], dim=-1)
        return y_t, next_state

    def single_step(self, dA, dB, C, u_t, state) -> torch.Tensor:
        """" Implementation using element-wise mul """
        # dA * state: (dA_re + i dA_im) * (state_re + i state_im)
        z_re = dA[..., 0] * state[..., 0] - dA[..., 1] * state[..., 1]
        z_im = dA[..., 1] * state[..., 0] + dA[..., 0] * state[..., 1]
        # Add input: u is real, so add u*(dB_re + i dB_im)
        u = u_t.unsqueeze(-1)  # [B, d_model, 1]
        u_re = u * dB[..., 0]
        u_im = u * dB[..., 1]
        # Update state
        next_state_re = z_re + u_re
        next_state_im = z_im + u_im
        # Output: y = 2 * Re( sum_n C_n * next_state )
        y_t = self.const2 * ((next_state_re * C[..., 0]).sum(dim=-1) - (next_state_im * C[..., 1]).sum(dim=-1))
        next_state = torch.stack([next_state_re, next_state_im], dim=-1)
        return y_t, next_state

    def step(self, u_t: torch.Tensor, state: torch.Tensor):
        return self.single_step(self.dA, self.dB, self.C, u_t, state)


class S4Layer(nn.Module):
    def __init__(
        self,
        d_model: int,
        T: int = 21,
        N: int = 64,
        dt_min: float = 0.66,
        dt_max: float = 1.0,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.ssm = S4D(d_model, T=T, N=N, dt_min=dt_min, dt_max=dt_max)
        self.pad_left = T-1 if causal else (T - 1) // 2
        self.pad_right = T-1 - self.pad_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        kernel = self.ssm.build_kernel(self.ssm.dt, self.ssm.A, self.ssm.B, self.ssm.C)          # [d_model, 1, T]
        x_T = x.transpose(1, 2)
        x_padded = F.pad(x_T, (self.pad_left, self.pad_right))
        y_T = F.conv1d(x_padded, kernel, groups=self.d_model)        # [B, d_model, T]
        y = y_T.transpose(1, 2)                                      # [B, T, d_model]
        return y

    def step(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.ssm.step(x, state)


class S4Block(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int,
        T: int = 21,
        N: int = 64,
        dt_min: float = 0.66,
        dt_max: float = 1.0,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.s4 = S4Layer(d_model, T=T, N=N, dt_min=dt_min, dt_max=dt_max, causal=causal)
        self.act = nn.GELU()
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        if d_in != d_model:
            self.skip_proj = nn.Linear(d_in, d_model)
        else:
            self.skip_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.in_proj(x)
        h = self.s4(h)
        h = self.act(h)
        h = self.out_proj(h)
        h = self.dropout(h)
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)
        y_t = self.norm(h + residual)
        return y_t

    def step(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.in_proj(x)
        h, new_state = self.s4.step(h, state)
        h = self.act(h)
        h = self.out_proj(h)
        if self.skip_proj is not None:
            residual = self.skip_proj(residual)
        y_t = self.norm(h + residual)
        return y_t, new_state


class S4VAD(nn.Module):
    """
    Voice Activity Detector with DSS-based architecture.
    Input:  [B, T, feat_dim]  (e.g. log-mel features)
    Output: frame-level probability of voice: [B, T]
    """
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 128,
        num_layers: int = 4,
        T: int = [21, 21, 21, 21],
        N: int = 64,
        dt_min: float = [0.6, 0.6, 0.6, 0.6],
        dt_max: float = [1.0, 1.0, 1.0, 1.0],
        dropout: float = 0.0,
        causal: bool = False,
        streaming: bool = False,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.d_model = d_model
        self.N2 = N//2
        self.num_layers = num_layers
        layers = []
        d_in = feat_dim
        for i in range(num_layers):
            layers.append(
                S4Block(
                    d_in=d_in,
                    d_model=d_model,
                    T=T[i],
                    N=N,
                    dt_min=dt_min[i],
                    dt_max=dt_max[i],
                    dropout=dropout,
                    causal=causal,
                )
            )
            d_in = d_model
        self.layers = nn.ModuleList(layers)
        self.head = nn.Linear(d_model, 1)
        self.final_norm = nn.LayerNorm(d_model)
        self.streaming = streaming

    def forward(self, x: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        if self.streaming:
            """
            streaming step.
            """
            assert x.shape[1] == 1, "second dim of frame should be 1"
            h = x.squeeze(1)
            next_states = []
            for l,layer in enumerate(self.layers):
                h, next_state = layer.step(h, state[:, l, ...])
                next_states.append(next_state)
            h = self.final_norm(h)
            logits = self.head(h).squeeze(-1)
            return logits, torch.stack(next_states, dim=1)
        else:
            """
            offline mode.
            """
            h = x
            for layer in self.layers:
                h = layer(h)

            h = self.final_norm(h)
            logits = self.head(h).squeeze(-1)  # [B, T]
            return logits


@register("model", "s4vad")
class S4VADAdapter(VADAdapter):
    def __init__(self, arch_cfg, sample_rate, segment_ms, streaming, on_threshold, off_threshold,
                 hard_decision, median_kernel_size, mel_cfg, melspec, preprocess, path=None):
        super().__init__(sample_rate=sample_rate, segment_ms=segment_ms, streaming=streaming, on_threshold=on_threshold,
                         off_threshold=off_threshold, hard_decision=hard_decision, median_kernel_size=median_kernel_size,
                         mel_cfg=mel_cfg, melspec= melspec, preprocess=preprocess)
        self.arch_cfg = arch_cfg
        self.model = S4VAD(**arch_cfg, streaming=streaming)
        self.load_weights(path)
        print(f"Total params and buffers: {(sum(p.numel() for p in self.model.parameters()) +
                                            sum(b.numel() for b in self.model.buffers()))/1000:.2f} K\n")

    def state_init(self, batch_size = 1, device = torch.device("cpu")) -> List[torch.Tensor]:
        return torch.zeros((batch_size,
                            self.arch_cfg["num_layers"],
                            self.arch_cfg["d_model"],
                            self.arch_cfg["N"]//2,
                            2), device=device)

    def postprocess(self, logits: torch.Tensor):
        probs = torch.sigmoid(logits)
        probs = probs.cpu().numpy()
        if self.streaming:
            probs = causal_median_filter(probs, kernel_size=self.median_kernel_size)
        else:
            probs = median_filter(probs, size=(self.median_kernel_size,1))
        return probs

    def stream_eval(self, batch: torch.Tensor):
        B, T, _ = batch.shape
        states = self.state_init(B, device=self.device)
        # Streaming
        logits = []
        for i in range(T):
            # Get current frame
            frame = batch[:,i:i+1,:]
            # Step
            with torch.inference_mode():
                logit, states = self.model(frame, states)
                logits.append(logit)
        logits = torch.stack(logits, dim=1)
        # Postprocess
        probs = self.postprocess(logits[:,-1])
        return np.stack(probs, axis=0)

    def stream_infer(self, wav: np.ndarray):
        # Padding
        pad_len = 1000
        wav = np.pad(wav, ((0, 0), (pad_len*self.hop_length, 0)), mode='reflect')
        wav_len = wav.shape[1]

        # In streaming, we do normalization online so we remove last transform of std_normalization
        self.transform = Compose(self.transform.transforms[:-1])

        # # Warmup state and online normalizer
        offset = 0
        state = self.state_init(1, device=self.device)
        onlineNormalizer = OnlineNormalizer(pad_len)
        while offset < pad_len*self.hop_length - self.n_fft:
            frame = wav[:, offset:offset + self.n_fft]
            # Preprocess
            frame = self.preprocess(frame)
            frame = onlineNormalizer.normalize(frame)
            # Streaming
            with torch.inference_mode():
                logit, state = self.model(frame, state)
            offset += self.hop_length

        # Streaming
        offset = pad_len*self.hop_length
        frames, logits, probs = [], [], []
        while offset < wav_len - self.n_fft:
            frame = wav[:, offset:offset + self.n_fft]
            # Preprocess
            frame = self.preprocess(frame)
            frame = onlineNormalizer.normalize(frame)
            frames.append(frame)
            # Streaming
            with torch.inference_mode():
                logit, state = self.model(frame, state)
            logits.append(logit)
            # Postprocess
            prob = self.postprocess(logit)
            probs.append(prob)
            offset += self.hop_length

        batch = torch.concat(frames, dim=0)
        logits = torch.concat(logits, dim=0)
        probs = np.stack(probs, axis=0)
        return probs, logits, batch
