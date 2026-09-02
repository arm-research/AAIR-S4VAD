# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import torch, torchaudio
import numpy as np
import random
import os

from matplotlib import pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def hz2mel(hz):
    """Convert a value in Hertz to Mels"""
    return 2595 * np.log10(1 + hz / 700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz"""
    return 700 * (10 ** (mel / 2595.0) - 1)

def plot_waveform(waveform, sample_rate=16000, title="waveform"):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        axes[c].set_xlabel('Time[sec]')
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.savefig(f"{title}.png")

def plot_spectogram(waveform, sample_rate=16000, title="Spectrogram"):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        axes[c].grid(True)
        axes[c].set_xlabel('Time[sec]')
        axes[c].set_ylabel('Freq[Hz]')
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle(title)
    plt.savefig(f"{title}.png")

def plot_frequency(waveform, sample_rate=16000, title="Frequency"):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].psd(waveform[c], Fs=sample_rate)
        axes[c].grid(True)
        axes[c].set_xlabel('Freq[Hz]')
        axes[c].set_ylabel('Amp')
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle(title)
    plt.savefig(f"{title}.png")

def plot_waveform_and_label(waveform, label, sample_rate=16000, title="waveform"):
    assert len(waveform.shape)<=2, "waveform dimensions are too much ! (no more than 2)"
    if len(waveform.shape)==1:
        waveform = waveform.unsqueeze(0)
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].plot(time_axis, label, linewidth=1, color="red")
        axes[c].grid(True)
        axes[c].set_xlabel('Time[sec]')
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.savefig(f"{title}.png")