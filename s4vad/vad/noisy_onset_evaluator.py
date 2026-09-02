# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import torch
from s4vad.common.adapters import DatasetAdapter, ModelAdapter
from s4vad.common.registries import register
from s4vad.common.metrics import auroc_score
from utils import get_device
from tqdm import tqdm
import numpy as np



@register("evaluator", "noisy_onset_evaluator")
class VadEvaluator(object):
    def __init__(self, deltaL=10, deltaR=10):
        self.deltaL = deltaL
        self.deltaR = deltaR
        self.off_interval, self.on_interval = 30, 30
        self.onsets = []

    @classmethod
    def build_from_config(cls, cfg: Dict):
        return cls(**cfg) if cfg else cls()

    @staticmethod
    def onset_score(p, ti_star, eps=1e-6):
        """Unnormalized score proportional to P(t* = t_star | x) from per-frame speech probs p."""
        K = len(p)
        if ti_star < 0 or ti_star >= K:
            assert False, "Error"
        p = np.clip(p, eps, 1 - eps)
        ps = np.prod(1 - p[:ti_star]) * p[ti_star]
        ps = np.clip(ps, eps, 1)
        return ps

    def qi_delta(self, p, ti, pi, deltas):
        """Compute q_i(delta) for all deltas for one sample i."""
        scores = np.array([self.onset_score(p, ti - d) for d in deltas], dtype=float)
        unqi = pi * scores
        qi = unqi / unqi.sum()
        return qi

    def em_noise_estimation(self, alpha=0.0, iterations=5):
        D = len(self.onsets)
        if D == 0:
            return None

        deltas = np.arange(-self.deltaL, self.deltaR + 1)
        m = len(deltas)
        pi = np.ones(m) / m  # init uniform
        denom = m * alpha + D  # constant each iteration
        for _ in range(iterations):
            # E-step: accumulate expected counts for each delta
            pi_iter = np.zeros(m)
            for onset, ti in self.onsets:
                pi_iter += self.qi_delta(onset, ti, pi, deltas)
            # M-step: update pi distribution
            pi = (alpha + pi_iter) / denom
        print(f"Label Noise estimation:")
        print(f"delta={deltas}")
        print(f"pi={pi}")
        # Calc expectation and std
        pi_mean = np.sum(pi * deltas)
        pi_sigma = np.sqrt(np.sum(pi * (deltas - pi_mean) ** 2))
        return pi_mean, pi_sigma

    def collect_onsets(self, target, prob):
        N = len(target)
        # Collect samples of onsets
        for n in range(self.off_interval, N):
            if np.sum(target[n:n+self.on_interval]) == self.on_interval and np.sum(1-target[n-self.off_interval:n])==self.off_interval:
                ti = n
                # ----------------------------------------
                # Add gaussian noise to ti
                # mu, sigma = -4, 3
                # ti = np.round(ti + np.random.normal(mu, sigma)).astype(int)
                # ----------------------------------------
                prob_window = prob[ti-self.off_interval : ti+self.on_interval]
                ti_window = self.off_interval
                onset = (prob_window, ti_window)
                self.onsets.append(onset)


    def evaluate(self, ma, dataloader, desc="Validation"):

        device = get_device()
        # Iterate Over Validation Dataloader to check for accuracy
        targets, probs = [], []
        for x_batch, y_true in tqdm(dataloader, desc=desc):
            if isinstance(x_batch, list):
                x_batch = torch.concat([item for item in x_batch])
                y_true = torch.concat([item for item in y_true])
            # Put batch on device
            x_batch = x_batch.to(device)
            y_true = y_true.to(device)
            # Forward Pass
            y_true = y_true[:, -1]
            y_prob = ma.stream_eval(x_batch)
            self.collect_onsets(y_true.cpu().numpy(), y_prob)
            targets.extend(y_true.cpu().numpy().flatten())
            probs.extend(y_prob.flatten())

        # AUROC
        auroc = auroc_score(targets, probs)
        print("AUROC: {:.3f}".format(auroc))

        # Onset Latency
        onset_noise_stat = self.em_noise_estimation()
        if onset_noise_stat is None:
            print("No speech onsets found; skipping label-noise estimation.")
            return auroc, None
        pi_mean, pi_sigma = onset_noise_stat
        print("Label Error mean: {:.3f}, std: {:.3f} for {} onsets".format(pi_mean, pi_sigma, len(self.onsets)))

        return auroc, onset_noise_stat

    def __call__(self, ma: ModelAdapter, da: DatasetAdapter):
        return self.evaluate(ma, da.val_loader, desc="Validation")
