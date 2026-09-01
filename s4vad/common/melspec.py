from typing import Optional
import math
import torch, torchaudio


class AmplitudeToDB(torch.nn.Module):
    def __init__(self, stype: str = "power", top_db: Optional[float] = None) -> None:
        super(AmplitudeToDB, self).__init__()
        self.stype = stype
        if top_db is not None and top_db < 0:
            raise ValueError("top_db must be positive value")
        self.register_buffer('top_db', torch.tensor(0 if top_db is None else top_db))
        amin = 1e-10
        ref_value = 1.0
        self.register_buffer('multiplier', torch.tensor(10.0 if stype == "power" else 20.0))
        self.register_buffer('amin', torch.tensor(amin))
        self.register_buffer('ref_value', torch.tensor(ref_value))
        self.register_buffer('db_multiplier', torch.tensor(math.log10(max(amin, ref_value))))
        self.top_db_func = torch.nn.Identity() if top_db is None else self._top_db

    def _top_db(self, x_db):
        # Expand batch
        shape = x_db.size()
        packed_channels = shape[-3] if x_db.dim() > 2 else 1
        x_db = x_db.reshape(-1, packed_channels, shape[-2], shape[-1])

        x_db = torch.max(x_db, (x_db.amax(dim=(-3, -2, -1)) - self.top_db).view(-1, 1, 1, 1))

        # Repack batch
        x_db = x_db.reshape(shape)
        return x_db

    def forward(self, x):
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier
        x_db = self.top_db_func(x_db)
        return x_db


class LogNorm(torch.nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super(LogNorm, self).__init__()
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, x):
        return (x + self.eps).log()


class LogMelConv(torch.nn.Module):
    def __init__(self,
                 sample_rate=16000,
                 n_fft=512,
                 n_mels=40,
                 hop_length=160,
                 win_length=480,
                 f_min = 0,
                 f_max = 8000,
                 center=True,
                 window_fn=torch.hann_window,
                 eps=1e-6,
                 norm_mode='log_norm'):
        super(LogMelConv, self).__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.k = n_fft//2 + 1
        self.n_mels = n_mels
        self.center = center
        self.input_pad = int(n_fft // 2) if center else 0
        self.window = window_fn(self.win_length)
        self.dft_conv = torch.nn.Conv1d(1, 2*(n_fft//2 + 1), n_fft, stride=hop_length, padding=0,
                                        bias=False)
        n = torch.arange(win_length).reshape(1, -1)
        k = n[0, :n_fft//2 + 1].reshape(-1, 1)
        real_weight = torch.cos(-2 * torch.pi * n * k / n_fft)
        imag_weight = torch.sin(-2 * torch.pi * n * k / n_fft)
        dft_weight = torch.cat([real_weight, imag_weight], dim=0)
        dft_weight = dft_weight * self.window.reshape(1, -1)
        pad_size = (n_fft - win_length) / 2
        assert pad_size.is_integer(), "n_fft - win_length must be even."
        pad_shape = (dft_weight.shape[0], int(pad_size))
        fft_weight = torch.cat([torch.zeros(pad_shape, dtype=dft_weight.dtype),
                                dft_weight,
                                torch.zeros(pad_shape, dtype=dft_weight.dtype)], dim=1)
        fft_weight = fft_weight.unsqueeze(1)
        assert fft_weight.shape == self.dft_conv.weight.shape
        self.dft_conv.weight.data = fft_weight
        self.dft_conv.weight.requires_grad_(False)
        self.register_module('mel_scale', torchaudio.transforms.MelScale(n_mels=n_mels,
                                                        sample_rate=sample_rate,
                                                        f_min=f_min,
                                                        f_max=f_max,
                                                        n_stft=self.k))

        if norm_mode == 'log_norm':
            self.norm_func = LogNorm(eps=eps)
        elif norm_mode == 'amplitude_to_db':
            # self.register_module('norm_func', AmplitudeToDB(top_db=80))
            self.register_module('norm_func', AmplitudeToDB(top_db=None))
        elif norm_mode == 'none':
            self.norm_func = torch.nn.Identity()
        else:
            raise ValueError("norm_mode must be 'log_norm' or 'amplitude_to_db'")

        self.register_buffer('norm_eps', torch.tensor(1e-8))

    def forward(self, x):
        x = torch.nn.functional.pad(x.unsqueeze(1), [self.input_pad, self.input_pad], "reflect")
        x = self.dft_conv(x)
        x_2 = torch.square(x)
        x_2_r, x_2_i = torch.split(x_2, self.k, dim=1)
        spec = x_2_r + x_2_i
        mel_scale = self.mel_scale(spec)
        output = self.norm_func(mel_scale)
        output = (output - output.mean()) / (output.std() + self.norm_eps)
        return output
