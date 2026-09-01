import numpy as np
import torch.nn.functional as F
from scipy.ndimage import median_filter
from s4vad.common.registries import register
from s4vad.common.melspec import LogMelConv
from s4vad.vad.models.vad import VADAdapter
from s4vad.vad.utils import Transpose, causal_median_filter
import torch
import torch.nn as nn


class DepthWiseConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 1,
                 stride: int = 1,
                 dilation: int =1,
                 causal: bool = False):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.padding = dilation * (kernel_size - 1) if causal else 0
        self.dw = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels,
                      kernel_size,
                      stride,
                      dilation=dilation,
                      groups=in_channels,
                      padding=0 if causal else dilation * (kernel_size - 1) //2,
                      bias=False),
            nn.Conv1d(in_channels, out_channels, 1, 1, bias=False))

    def forward(self, x):
        x = F.pad(x, (self.padding,0)) # padding for causal
        x = self.dw(x)
        return x


class SubBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 C: int,
                 kernel_size: int,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        self.dw = DepthWiseConv(in_channels, C, kernel_size=kernel_size, causal=causal)
        self.bn = nn.BatchNorm1d(C)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.dw(x))))


class ConvBlock(nn.Module):
    def __init__(self,
                 R: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dropout: float = 0.1,
                 causal: bool = False):
        super().__init__()
        self.block_list = nn.ModuleList()

        subblock_channels = in_channels
        for _ in range(R):
            subblock = SubBlock(subblock_channels, out_channels, kernel_size, dropout, causal)
            subblock_channels = out_channels
            self.block_list.append(subblock)

        self.pw_res = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.bn_res = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout1d(p=dropout)

    def forward(self, x):
        residual = x

        # Main block
        for block in self.block_list:
            x = block(x)

        # Residual
        residual = self.bn_res(self.pw_res(residual))
        x = x + residual

        x = self.dropout(F.relu(x))
        return x


class MarbleNet(nn.Module):
    def __init__(self,
                 in_channels: int = 64,
                 out_channels: int = 128,
                 R: int = 2,
                 C: int = 64,
                 kernel_size_blockwise: list = (13, 15, 17),
                 dropout: float = 0.1,
                 causal: bool = False,
                 melspec='none',
                 mel_cfg=None,
                 sample_rate=16000):
        super().__init__()
        # Front
        self.front = nn.Sequential(DepthWiseConv(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=11,
                                                 causal=causal),
                                   nn.BatchNorm1d(out_channels),
                                   nn.ReLU())


        assert melspec in ['none', 'conv'], f"unknown melspec mode '{melspec}'"
        self.melspec = melspec
        if self.melspec == 'conv':
            assert mel_cfg is not None, "mel_cfg must be provided."
            self.register_module('preprocess', LogMelConv(sample_rate, **mel_cfg))
        else:
            self.preprocess = Transpose(1,2)

        # Blocks
        self.blocks = nn.ModuleList()

        in_channels = out_channels
        for kernel_size in kernel_size_blockwise:
            block = ConvBlock(R,
                              in_channels,
                              out_channels=C,
                              kernel_size=kernel_size,
                              dropout=dropout,
                              causal=causal)
            in_channels = C
            self.blocks.append(block)

        # Epilogue
        self.conv2 = nn.Sequential(
            DepthWiseConv(in_channels=C,
                      out_channels=out_channels,
                      kernel_size=29,
                      dilation=2,
                      causal=causal),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=out_channels,
                      out_channels=out_channels,
                      kernel_size=1,
                      stride=1,
                      bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
        self.linear = nn.Linear(out_channels, 1)

    def forward(self, x):
        x = self.preprocess(x)
        x = self.front(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.linear(x.transpose(1, 2))
        x = x.squeeze(-1)
        return x


@register("model", "marblenet")
class MarbleNetAdapter(VADAdapter):
    def __init__(self, arch_cfg, sample_rate, segment_ms, streaming, on_threshold, off_threshold,
                 hard_decision, median_kernel_size, mel_cfg, melspec, preprocess, path=None):
        super().__init__(sample_rate=sample_rate, segment_ms=segment_ms, streaming=streaming, on_threshold=on_threshold,
                         off_threshold=off_threshold, hard_decision=hard_decision, median_kernel_size=median_kernel_size,
                         mel_cfg=mel_cfg, melspec= melspec, preprocess=preprocess)
        self.model = MarbleNet(**arch_cfg,
                               melspec=melspec,
                               mel_cfg=mel_cfg,
                               sample_rate=sample_rate)
        self.load_weights(path)
        print(f"Total params and buffers: {(sum(p.numel() for p in self.model.parameters()) +
                                            sum(b.numel() for b in self.model.buffers()))/1000:.2f} K\n")

    def postprocess(self, logits: torch.Tensor):
        probs = torch.sigmoid(logits)
        probs = probs.cpu().numpy()
        if self.streaming:
            probs = probs[:, -1]
            probs = causal_median_filter(probs, kernel_size=self.median_kernel_size)
        else:
            # Median filter
            probs = median_filter(probs, size=(self.median_kernel_size,1))
        return probs

    def stream_eval(self, batch: torch.Tensor):
        with torch.inference_mode():
            logits = self(batch)
        # Postprocess
        probs = self.postprocess(logits)
        return np.stack(probs, axis=0)

    def stream_infer(self, wav: np.ndarray):
        # Padding
        in_wav = np.pad(wav, ((0, 0), (self.segment_len - self.hop_length, 0)), mode='reflect')
        wav_len = in_wav.shape[1]
        frames, logits, probs = [], [], []
        for idx in range(0, wav_len - self.segment_len, self.hop_length):
            frame = in_wav[:, idx:idx + self.segment_len]
            # Preprocess
            frame = self.preprocess(frame)
            frames.append(frame)
            # Streaming
            with torch.inference_mode():
                logit = self(frame)
            logits.append(logit)
            # Postprocess
            prob = self.postprocess(logit)
            probs.append(prob)
        batch = torch.concat(frames, dim=0)
        logits = torch.concat(logits, dim=0)
        probs = np.stack(probs, axis=0)
        return probs, logits, batch
