import numpy as np
import torch
from matplotlib import pyplot as plt
from s4vad.common.registries import register
from s4vad.vad.models.vad import VADAdapter
from silero_vad import load_silero_vad, get_speech_timestamps


@register("model", "silero")
class SileroAdapter(VADAdapter):
    def __init__(self, sample_rate, on_threshold, off_threshold, hard_decision, mel_cfg, melspec, preprocess='none'):
        super().__init__(sample_rate=sample_rate, segment_ms=512, streaming=True, on_threshold=on_threshold,
                         off_threshold=off_threshold, hard_decision=hard_decision, median_kernel_size=1,
                         mel_cfg=mel_cfg, melspec= melspec, preprocess=preprocess)
        self.model = load_silero_vad().to(self.device)
        self.model.eval()
        print(f"Total params and buffers: {(sum(p.numel() for p in self.model.parameters()) +
                                            sum(b.numel() for b in self.model.buffers()))/1000:.2f} K\n")

    def preprocess(self, wav: np.ndarray):
        # Mono
        wav = wav[0:1,:]
        # Torch
        in_tensor = torch.as_tensor(wav, dtype=torch.float)
        in_tensor = self.transform(in_tensor)
        in_tensor = in_tensor.contiguous().to(self.device)
        return in_tensor

    def postprocess(self, x):
        """Post process the output of the model"""
        return x.squeeze().cpu().numpy()

    def stream_eval(self, batch: torch.Tensor):
        B, T, L = batch.shape
        probs = []
        for b in range(B):
            frames = batch[b,...]
            with torch.inference_mode():
                prob = self.model(frames, self.sample_rate)
            prob = self.postprocess(prob[-1,:])
            probs.append(prob)
        return np.stack(probs, axis=0)

    def stream_infer(self, batch: torch.Tensor):
        with torch.inference_mode():
            probs = self.model(batch, self.sample_rate)
        probs = self.postprocess(probs)
        return probs

    def forward(self, batch: torch.Tensor):
        with torch.inference_mode():
            model_out =  self.model(batch, self.sample_rate)
        return model_out

    def infer(self, wav: np.ndarray, input_sr: int = None, plot=True):
        # Torch
        in_tensor = torch.as_tensor(wav, dtype=torch.float)
        in_tensor = in_tensor.contiguous().to(self.device)
        speech_timestamps = get_speech_timestamps(
            in_tensor,
            self.model,
            threshold=self.on_threshold,
        )
        print(f"Speech timestamps: {speech_timestamps}")
        activity_samples = np.zeros((wav.shape[-1]))
        for timestamp in speech_timestamps:
            start, end = timestamp["start"], timestamp["end"]
            activity_samples[start:end] = 1

        wav = wav / np.abs(wav).max()
        sec = np.arange(0, len(activity_samples) / self.sample_rate, 1 / self.sample_rate)
        wav = wav[0, :len(sec)]
        activity_samples = activity_samples[:len(sec)]
        if plot:
            plt.plot(sec, wav)
            plt.plot(sec, activity_samples, color='red')
            plt.xlabel('Time [sec]')
            plt.ylabel('Activity')
            plt.savefig('vad.png')
        return wav, activity_samples
