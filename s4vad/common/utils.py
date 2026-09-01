from functools import partial
import numpy as np
import torch
import torchaudio
from torchvision.transforms import v2


def std_normalize(wav, eps=1e-8):
    return (wav - wav.mean()) / (wav.std() + eps)


def generate_mix_snr_noise(sig, noise, snr, eps=1e-8):
    Es = torch.mean(sig**2)
    En = torch.mean(noise**2)
    gain = torch.sqrt((Es/(En+eps))/(10**(snr/10))) if Es>0 else 1.0
    return sig + gain*noise


def spec_augment(time_mask_param=0, freq_mask_param=0, num_masks=1):
    transforms = []
    for _ in range(num_masks):
        transforms.extend([torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param),
                           torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)])
    transform = v2.Compose(transforms)
    return transform


def audio_transform(sample_rate, mel_cfg):
    transform = v2.Compose([torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, **mel_cfg),
                            partial(torch.squeeze, dim=0),
                            partial(torch.add, 1e-6),
                            partial(torch.log),
                            partial(torch.transpose, dim0=0, dim1=1),
                            partial(std_normalize)])
    return transform
