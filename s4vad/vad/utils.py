from functools import partial
import numpy as np
import torch
from torchvision.transforms import v2
from s4vad.common.melspec import LogMelConv
from scipy.ndimage import maximum_filter, minimum_filter, median_filter
import torch.nn.functional as F
from collections import deque

from s4vad.common.utils import audio_transform, spec_augment


MAX_AMP = 0.9

def torch_pad_wav_if_needed(wav, pad_size):
    if wav.shape[1] < pad_size:
        wav_padd = np.pad(wav.cpu().numpy(), ((0, 0), (0, pad_size - wav.shape[1])), mode='reflect')
        return torch.from_numpy(wav_padd)
    return wav


def torch_pad_label_if_needed(label, pad_size):
    if len(label) < pad_size:
        label_padd = np.pad(label.cpu().numpy(), (0, pad_size - len(label)), mode='reflect')
        return torch.from_numpy(label_padd)
    return label


class Transpose(torch.nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


def get_train_transform(sample_rate, preprocess, mel_cfg, time_mask_param=0, freq_mask_param=0):
    assert preprocess in ['none', 'fft', 'conv'], f"unknown preprocess mode '{preprocess}'"
    if preprocess == 'fft':
        assert mel_cfg is not None, "mel_cfg must be provided."
        transform = v2.Compose([audio_transform(sample_rate, mel_cfg),
                                spec_augment(time_mask_param, freq_mask_param)])
    elif preprocess == 'conv':
        assert mel_cfg is not None, "mel_cfg must be provided."
        transform = v2.Compose([LogMelConv(sample_rate, **mel_cfg),
                                partial(torch.squeeze, dim=0),
                                spec_augment(time_mask_param, freq_mask_param)])
    else:
        transform = partial(torch.squeeze, dim=0)
    return transform


def split_audio_to_chunks(
    audio: torch.Tensor,
    chunk_length: int,
    stride: int,
    *,
    pad: bool = False,
    pad_value: float = 0.0):
    """
    Split a 1D audio signal [:,N] into fixed-size chunks.
    """
    n = int(audio.shape[1])
    starts = list(range(0, n, stride))

    chunks, positions = [], []
    for s in starts:
        e = s + chunk_length
        if e <= n:
            chunk = audio[:,s:e]
            chunks.append(chunk)
            positions.append((s, e))
        else:
            # Incomplete last chunk
            if not pad:
                break
            tail = audio[:,s:n]
            pad_len = e - n
            chunk = F.pad(tail, (0, pad_len), mode="constant", value=pad_value)
            chunks.append(chunk)
            positions.append((s, e))
            break  # after padding the last chunk, we're done

    chunks_arr = torch.concat(chunks, dim=0) if chunks else torch.empty((0, chunk_length), dtype=audio.dtype)
    return chunks_arr


def get_val_transform(sample_rate, preprocess, mel_cfg):
    assert preprocess in ['none', 'fft', 'conv', 'split'], f"unknown preprocess mode '{preprocess}'"
    if preprocess == 'fft':
        assert mel_cfg is not None, "mel_cfg must be provided."
        transform = audio_transform(sample_rate, mel_cfg)
    elif preprocess == 'conv':
        assert mel_cfg is not None, "mel_cfg must be provided."
        transform = v2.Compose([LogMelConv(sample_rate, **mel_cfg), partial(torch.squeeze, dim=0)])
    elif preprocess == 'split':
        assert mel_cfg is not None, "mel_cfg must be provided."
        transform = partial(split_audio_to_chunks, chunk_length=mel_cfg.get("n_fft"), stride=mel_cfg.get("hop_length"))
    else:
        transform = partial(torch.squeeze, dim=0)
    return transform


def majority_vote(label, n_fft, hop_len, mode="median"):
    if mode=="max":
        label_smooth = maximum_filter(label, size=n_fft)
    elif mode=="min":
        label_smooth = minimum_filter(label, size=n_fft)
    elif mode=="median":
        label_smooth = median_filter(label, size=n_fft)
    else:
        label_smooth = label
    # frame centers (same convention as STFT)
    start = n_fft // 2
    frame_labels = label_smooth[start::hop_len]
    # trim to valid frames only
    n_frames = 1 + (len(label) - n_fft) // hop_len
    return frame_labels[:n_frames]


def causal_median_filter(probs, kernel_size):
    if probs.ndim == 1:
        probs = probs[:, None]  # [T, 1]
    # Pad ONLY on the left (past), not on the right
    pad = kernel_size - 1
    padded = np.pad(
        probs,
        pad_width=((pad, 0), (0, 0)),
        mode="edge"   # repeat earliest value
    )
    # Apply standard median filter
    filtered = median_filter(
        padded,
        size=(kernel_size, 1),
        mode="nearest"
    )
    # Remove padding
    causal_filtered = filtered[pad:, :]
    return causal_filtered.squeeze()


class OnlineNormalizer:
    def __init__(self, buffer_size: int, eps: float = 1e-8):
        self.buffer_size = buffer_size
        self.eps = eps
        self.mean_buffer = deque(maxlen=buffer_size)
        self.mean2_buffer = deque(maxlen=buffer_size)
        self.running_mean = 0
        self.running_var = 1

    def _update_running_stats(self):
        means = torch.stack(list(self.mean_buffer))
        mean2s = torch.stack(list(self.mean2_buffer))
        self.running_mean = means.mean()
        self.running_std = (mean2s.mean() - self.running_mean ** 2).sqrt()

    @torch.no_grad()
    def normalize(self, x: torch.Tensor):
        batch_mean = x.mean()
        batch_mean2 = (x ** 2).mean()
        # Update buffers
        self.mean_buffer.append(batch_mean)
        self.mean2_buffer.append(batch_mean2)
        self._update_running_stats()
        # Normalize using current stats
        x_norm = (x - self.running_mean) / (self.running_std + self.eps)
        return x_norm
