# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import json
from glob import glob
from typing import Dict, Any
import pandas as pd
from torchaudio import load
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

from s4vad.common.adapters import DatasetAdapter
from s4vad.common.registries import register
from s4vad.common.utils import generate_mix_snr_noise
from s4vad.vad.utils import get_val_transform, majority_vote, MAX_AMP, torch_pad_wav_if_needed
import torchaudio.functional as F


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


class AVASpeech(Dataset):
    def __init__(self, ds):
        self.path = ds.get("path")
        self.audios, self.labels = [], []
        self.sample_rate = ds.get("sample_rate")
        self.segment_len = int(self.sample_rate*ds.get("segment_ms")*1e-3)
        self.segment_hop_len = int(self.sample_rate*ds.get("segment_hop_ms")*1e-3)
        self.ava_split = ds.get("ava_split","dataset")
        mel_cfg = ds.get("mel_cfg", None)
        self.n_fft = mel_cfg.get("n_fft")
        self.hop_length = mel_cfg.get("hop_length")
        self.transform = get_val_transform(self.sample_rate,
                                           ds.get("preprocess","fft"),
                                           mel_cfg)
        self.snrs = ds.get("snrs",None)
        if self.snrs is not None:
            noise_path = ds.get("noise_path", None)
            assert noise_path is not None, "noise path must be provided"
            self.noise_samples = get_noise_samples(noise_path)
        self.max_samples = ds.get("max_samples", -1)
        self._prepare_dataset()

    def _prepare_dataset(self):
        label_mapping = {
            'NO_SPEECH': 0,
            'CLEAN_SPEECH': 1,
            'SPEECH_WITH_MUSIC': 1,
            'SPEECH_WITH_NOISE': 1
        }

        # Read label file
        df = pd.read_csv(os.path.join(self.path, "ava_speech_labels.csv"))
        df.columns = ["file", "start", "stop", "label"]
        df_group = df.groupby("file")

        # Run over groups and create sequences
        counter = 0
        for key, group in tqdm(df_group):
            audio, label = [], []
            for index, row in group.iterrows():
                folder_name = row.iloc[3]
                file_name = row.iloc[0]+"-"+str(row.iloc[1])+"-"+str(row.iloc[2])+".wav"
                file_path = os.path.join(self.path,self.ava_split,folder_name,file_name)
                if not os.path.exists(file_path):
                    continue
                speech = label_mapping.get(folder_name, -1)
                wav, _ = load(file_path)
                wav_label = speech * torch.ones(wav.shape[-1])
                audio.append(wav)
                label.append(wav_label)

            if len(audio) > 0:
                audio = torch.cat(audio, dim=-1)
                label = torch.cat(label, dim=-1)
                if self.snrs is not None:
                    num_samples = audio.shape[-1]
                    # Get noise sample
                    noise_sample = self.noise_samples[counter % len(self.noise_samples)]
                    noise, sr = load(noise_sample)
                    noise = F.resample(noise, orig_freq=sr, new_freq=self.sample_rate)
                    noise = noise[0:1, :num_samples]  # mono
                    noise = torch_pad_wav_if_needed(noise, num_samples)
                    # Mix with audio
                    audio = generate_mix_snr_noise(audio, noise, self.snrs[counter % len(self.snrs)])
                    counter += 1

                # Split to chunks (up to 1min)
                N = audio.shape[-1]
                chunk_size = min(N, self.sample_rate*60)
                n = 0
                while n <= N - chunk_size:
                    self.audios.append(audio[:,n:n+chunk_size])
                    self.labels.append(label[n:n+chunk_size])
                    n += chunk_size

    def __len__(self):
        if self.max_samples==-1:
            return len(self.audios)
        else:
            return min(self.max_samples, len(self.audios))

    def __getitem__(self, idx):
        audio = self.audios[idx]
        label = self.labels[idx]
        num_samples = audio.shape[-1]

        # Split to segments
        segments, targets = [], []
        for idx in range(0, num_samples - self.segment_len, self.segment_hop_len):
            audio_seg = audio[:, idx:idx + self.segment_len]
            # If it clips, normalize
            if torch.max(torch.abs(audio_seg)) >= 1:
                audio_seg = audio_seg * MAX_AMP / torch.max(torch.abs(audio_seg))
            label_seg = label[idx:idx + self.segment_len]
            # Target
            target = majority_vote(label_seg, self.n_fft, self.hop_length)
            targets.append(target)
            # Transform
            segment = self.transform(audio_seg)
            segments.append(segment)

        return segments, targets


@register("dataset", "avaspeech")
class AVASpeechAdapter(DatasetAdapter):
    def __init__(self, val_ds):
        self.val_ds = val_ds
        valset = AVASpeech(val_ds)
        print("Test Dataset Size: {}".format(len(valset)))
        self.val_loader = DataLoader(valset,
                                     batch_size=1,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=val_ds.get("num_workers", 0))


    @classmethod
    def build_from_config(cls, cfg: Dict[str, Any]):
        return cls(**cfg) if cfg else cls()
