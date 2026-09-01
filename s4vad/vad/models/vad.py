from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import torch
from matplotlib import pyplot as plt
from s4vad.common.adapters import ModelAdapter
from s4vad.vad.utils import get_val_transform
from utils import get_device


class VADAdapter(ModelAdapter, ABC):
    def __init__(self, sample_rate, segment_ms, streaming, on_threshold, off_threshold,
                 hard_decision, median_kernel_size, mel_cfg, melspec, preprocess):
        super().__init__()
        self.device = get_device()
        self.sample_rate = sample_rate
        self.segment_len = int(sample_rate * segment_ms * 1e-3)
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.hard_decision = hard_decision
        self.streaming = streaming
        self.melspec = melspec
        self.median_kernel_size = median_kernel_size
        self.mel_cfg = mel_cfg
        self.n_fft = self.mel_cfg.get("n_fft")
        self.hop_length = self.mel_cfg.get("hop_length")
        self.transform = get_val_transform(sample_rate, preprocess, mel_cfg)
        self.model = None

    @classmethod
    def build_from_config(cls, cfg: Dict):
        return cls(**cfg) if cfg else cls()

    def preprocess(self, wav: np.ndarray):
        # Mono
        wav = wav[0:1,:]
        # Torch
        in_tensor = torch.as_tensor(wav, dtype=torch.float)
        in_tensor = self.transform(in_tensor)
        in_tensor = in_tensor.unsqueeze(0) # add batch dim
        in_tensor = in_tensor.contiguous().to(self.device)
        return in_tensor

    @abstractmethod
    def postprocess(self, *args, **kwargs):
        """Post process the output of the model"""

    @abstractmethod
    def stream_eval(self, *args, **kwargs):
        """Stream mode for evaluation"""

    @abstractmethod
    def stream_infer(self, *args, **kwargs):
        """Stream mode for inference"""

    def forward(self, batch: torch.Tensor):
        with torch.inference_mode():
            model_out =  self.model(batch)
        return model_out

    def prob2sample(self, probs):
        activity = []
        speech = 0
        for prob in probs:
            if prob > self.on_threshold and not speech:
                value = 1 if self.hard_decision else prob
                activity.extend(self.hop_length*[value])
                speech = 1
            elif prob < self.off_threshold and speech:
                value = True if self.hard_decision else prob
                activity.extend(self.hop_length*[value])
                speech = 0
            else:
                value = speech if self.hard_decision else prob
                activity.extend(self.hop_length*[value])
        return np.array(activity)

    def infer(self, wav: np.ndarray, input_sr: int = None, plot=True):
        if self.streaming:
            probs, logits, batch = self.stream_infer(wav)
        else:
            # Preprocess
            batch = self.preprocess(wav)
            # Infer
            logits = self(batch)
            # Postprocess
            probs = self.postprocess(logits)
            probs = probs.squeeze(0)  # remove batch size

        wav = wav / np.abs(wav).max()
        activity_samples = self.prob2sample(probs)
        sec = np.arange(0, len(activity_samples) / self.sample_rate, 1 / self.sample_rate)
        wav = wav[0, :len(sec)]
        activity_samples = activity_samples[:len(sec)]

        if plot:
            plt.plot(sec, wav)
            plt.plot(sec, activity_samples, color='red')
            plt.xlabel('Time [sec]')
            plt.ylabel('Activity')
            plt.savefig('vad.png')
            # Save test-vectors
            np.savez("test_vectors.npz",
                     audio_input=wav,
                     model_input=batch.cpu().numpy(),
                     model_output=logits.cpu().numpy(),
                     post_process=probs)

        return wav, activity_samples

    def load_weights(self, path):
        if path is not None:
            sd = torch.load(path, map_location="cpu")
            self.model.load_state_dict(sd)
        self.model.eval()
        self.model.to(self.device)
