import json
from typing import Dict, Any
import numpy as np
import torchaudio.functional as F
from s4vad.common.registries import register
from s4vad.common.adapters import DatasetAdapter
from torch.utils.data import DataLoader, Dataset
import os
from torchaudio import load
from s4vad.common.utils import generate_mix_snr_noise
from s4vad.vad.utils import torch_pad_wav_if_needed, get_val_transform, get_train_transform, \
    torch_pad_label_if_needed, majority_vote, MAX_AMP
import torch
from glob import glob
import torch.nn as nn




class Augmentor(nn.Module):
    def __init__(self, snr_range=[0,10], white_noise_range=[-90,-46]):
        super().__init__()
        self.snr_range = snr_range
        self.white_noise_range = white_noise_range

    def add_white_noise(self, wav):
        noise_level_db = np.random.randint(self.white_noise_range[0], self.white_noise_range[1], dtype='int32')
        noise_signal = torch.randn_like(wav) * (10.0 ** (noise_level_db / 20.0))
        wav += noise_signal
        return wav

    def forward(self, speech: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # Random SNR
        snr = np.random.uniform(low=self.snr_range[0], high=self.snr_range[1])
        speech = generate_mix_snr_noise(speech, noise, snr)

        # Add white noise
        speech = self.add_white_noise(speech)

        return speech

def get_clean_samples(path):
    clean_samples = []
    with open(path) as json_file:
        md = json.load(json_file)
        for sample in md:
            starts, stops, words = sample['VAD']['start'], sample['VAD']['stop'], sample['VAD']['word']
            if len(starts)==0:
                continue
            clean_sample = {"path": sample["origin_path"], "starts": starts, "stops": stops, "words": words}
            clean_samples.append(clean_sample)
    return clean_samples

def get_noise_samples(paths):
    if not isinstance(paths, list):
        paths = [paths]
    noise_samples = []
    for path in paths:
        if path.endswith(".json"):
            with open(path) as json_file:
                md = json.load(json_file)
                noise_samples.extend([item['path'] for item in md])
        else:
            # Save all wav files in folder
            for file in sorted(glob(os.path.join(path, "**", "*.wav"), recursive=True)):
                noise_samples.append(file)
    return noise_samples

def labeler(clean, starts, stops, sample_rate=16000, rand=True):
    speech, label = [], []

    # Add silence
    silence = np.random.randint(sample_rate, 3*sample_rate) if rand else 4*sample_rate
    speech.append(torch.zeros((1,silence)))
    label.append(torch.zeros(silence))

    # Add speech
    for start, stop in zip(starts, stops):
        clean_segment = clean[:,int(sample_rate * start):int(sample_rate * stop)]
        speech.append(clean_segment)
        label.append(torch.ones(clean_segment.shape[-1]))

    # Add silence
    silence = np.random.randint(sample_rate, 3*sample_rate) if rand else 4*sample_rate
    speech.append(torch.zeros((1,silence)))
    label.append(torch.zeros(silence))

    speech = torch.cat(speech,dim=-1)
    label = torch.cat(label,dim=-1)
    return speech, label


class LibrispeechTrain(Dataset):
    def __init__(self, ds):
        self.clean_samples = get_clean_samples(ds.get("path"))
        self.noise_samples = get_noise_samples(ds.get("noise_path"))
        self.sample_rate = ds.get("sample_rate", 16000)
        self.segment_len = int(self.sample_rate * ds.get("segment_ms", 630)*1e-3)
        self.n_fft = ds.get("mel_cfg").get("n_fft")
        self.hop_length = ds.get("mel_cfg").get("hop_length")
        aug = ds.get("aug_cfg", {})
        self.aug = Augmentor(snr_range=aug.get("snr_range",[-5,20]))
        self.transform = get_train_transform(self.sample_rate,
                                             ds.get("preprocess","fft"),
                                             mel_cfg = ds.get("mel_cfg", None),
                                             time_mask_param=aug.get("time_mask_param", 0),
                                             freq_mask_param=aug.get("freq_mask_param", 0))

    def __len__(self):
        return max(len(self.clean_samples), len(self.noise_samples))

    def __getitem__(self, idx):
        # Get speech sample
        clean_sample = self.clean_samples[idx % len(self.clean_samples)]
        path, starts, stops = clean_sample["path"], clean_sample['starts'], clean_sample['stops']
        clean, _ = load(path)
        speech, label = labeler(clean, starts, stops, sample_rate=self.sample_rate)
        torch_pad_wav_if_needed(speech, self.segment_len)
        torch_pad_label_if_needed(label, self.segment_len)

        # Take segment
        frame_start = np.random.randint(0, max(1,speech.shape[1]-self.segment_len))
        speech = speech[0:1,frame_start:frame_start+self.segment_len] # mono
        speech = torch_pad_wav_if_needed(speech, self.segment_len)
        label = label[frame_start:frame_start+self.segment_len]
        label = torch_pad_label_if_needed(label, self.segment_len)

        # Get noise sample
        noise_sample = self.noise_samples[idx % len(self.noise_samples)]
        noise, sr = load(noise_sample, num_frames=3*self.segment_len)
        noise = F.resample(noise, orig_freq=sr, new_freq=self.sample_rate)
        noise = noise[0:1,:self.segment_len] # mono
        noise = torch_pad_wav_if_needed(noise, self.segment_len)

        # Augmentation
        segment = self.aug(speech, noise)
        # If it clips, normalize
        if torch.max(torch.abs(segment)) >= 1:
            segment = segment * MAX_AMP / torch.max(torch.abs(segment))

        target = majority_vote(label, self.n_fft, self.hop_length)

        # Transform
        segment = self.transform(segment)

        return segment, target


class LibrispeechVal(Dataset):
    def __init__(self, ds):
        self.clean_samples = get_clean_samples(ds.get("path"))
        self.noise_samples = get_noise_samples(ds.get("noise_path"))
        self.sample_rate = ds.get("sample_rate", 16000)
        self.segment_len = int(self.sample_rate * ds.get("segment_ms", 630)*1e-3)
        self.segment_hop_len = int(self.sample_rate * ds.get("segment_hop_ms", 630)*1e-3)
        self.n_fft = ds.get("mel_cfg").get("n_fft")
        self.hop_length = ds.get("mel_cfg").get("hop_length")
        self.snrs = ds.get("snrs")
        self.transform = get_val_transform(self.sample_rate,
                                           ds.get("preprocess","fft"),
                                           ds.get("mel_cfg", None))
        self.max_samples = ds.get("max_samples", -1)

    def __len__(self):
        if self.max_samples==-1:
            return max(len(self.clean_samples), len(self.noise_samples))
        else:
            return self.max_samples

    def __getitem__(self, idx):
        # Get speech sample
        clean_sample = self.clean_samples[idx % len(self.clean_samples)]
        path, starts, stops = clean_sample["path"], clean_sample['starts'], clean_sample['stops']
        clean, _ = load(path)
        speech, label = labeler(clean, starts, stops, sample_rate=self.sample_rate, rand=False)
        torch_pad_wav_if_needed(speech, self.segment_len)
        torch_pad_label_if_needed(label, self.segment_len)
        num_samples = speech.shape[-1]

        # Get noise
        noise_sample = self.noise_samples[idx % len(self.noise_samples)]
        noise, sr = load(noise_sample, num_frames=3*num_samples)
        noise = F.resample(noise, orig_freq=sr, new_freq=self.sample_rate)
        noise = noise[0:1,:num_samples] # mono
        noise = torch_pad_wav_if_needed(noise, num_samples)

        # Mix
        noisy_speech = generate_mix_snr_noise(speech, noise, self.snrs[idx % len(self.snrs)])
        # Add gaussian Noise
        noisy_speech += torch.randn_like(noisy_speech) * 0.01
        # If it clips, normalize
        if torch.max(torch.abs(noisy_speech)) >= 1:
            noisy_speech = noisy_speech * MAX_AMP / torch.max(torch.abs(noisy_speech))

        # Split into segments
        segments, targets = [], []
        start = 0
        while start + self.segment_len <= num_samples:
            # Segment
            segment = noisy_speech[:,start:start + self.segment_len]
            segment = torch_pad_wav_if_needed(segment, self.segment_len)
            # Transform
            segment = self.transform(segment)
            segments.append(segment)
            # Target
            target = majority_vote(label[start:start + self.segment_len], self.n_fft, self.hop_length)
            targets.append(target)
            start += self.segment_hop_len

        return segments, targets

@register("dataset", "librispeech")
class LibrispeechAdapter(DatasetAdapter):
    def __init__(self, train_ds=None, val_ds=None):
        if train_ds is not None:
            self.train_ds = train_ds
            trainset = LibrispeechTrain(train_ds)
            print("Training Dataset Size: {}".format(len(trainset)))
            self.train_loader = DataLoader(trainset,
                                           batch_size=train_ds.get("batch_size", 32),
                                           shuffle=True,
                                           drop_last=True,
                                           num_workers=train_ds.get("num_workers", 4))

        if val_ds is not None:
            self.val_ds = val_ds
            valset = LibrispeechVal(val_ds)
            print("Validation Dataset Size: {}".format(len(valset)))
            self.val_loader = DataLoader(valset,
                                         batch_size=1,
                                         shuffle=False,
                                         drop_last=False,
                                         num_workers=val_ds.get("num_workers", 4))


    @classmethod
    def build_from_config(cls, cfg: Dict[str, Any]):
        return cls(**cfg) if cfg else cls()

