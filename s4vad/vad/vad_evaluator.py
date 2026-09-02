# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
from typing import Dict
import torch
from s4vad.common.adapters import DatasetAdapter, ModelAdapter
from s4vad.common.registries import register
from s4vad.common.metrics import auroc_score, f1_acc_score
from utils import get_device
from tqdm import tqdm
import numpy as np




@register("evaluator", "vad_evaluator")
class VadEvaluator(object):
    def __init__(self, streaming=True, K=5, target_fpr=0.03, target_tau=None, onset_error_bias=0):
        self.streaming = streaming
        self.K = K
        self.target_fpr = target_fpr
        self.target_tau = target_tau
        self.onset_error_bias = onset_error_bias
        self.thresholds = np.linspace(0.0, 1.0, num=500, endpoint=False)
        self.metrics_th = np.zeros((len(self.thresholds),4))
        self.off_interval, self.on_interval = 100, 100
        self.start_interval = 0
        self.onsets = []
        self.biases_th = [[] for _ in self.thresholds]

    @classmethod
    def build_from_config(cls, cfg: Dict):
        return cls(**cfg) if cfg else cls()

    def get_onset_metric(self, targets, probs):
        biases, stds, taus, nums, misses = [], [], [], [], []

        fprs = [f1_acc_score(targets, probs, tau)[2] for tau in self.thresholds]
        for tau_idx, tau in enumerate(self.thresholds):
            num = self.metrics_th[tau_idx][0]
            miss = self.metrics_th[tau_idx][3]
            if num == 0:
                biases.append(0)
                stds.append(0)
                taus.append(tau)
                nums.append(0)
                misses.append(0)
                continue
            if num > miss:
                bias = self.metrics_th[tau_idx][1] / (num - miss)
                mean2 = self.metrics_th[tau_idx][2] / (num - miss)
            else:
                bias, mean2 = 0, 0
            std = np.sqrt(mean2 - bias**2)
            biases.append(bias)
            stds.append(std)
            taus.append(tau)
            nums.append(num)
            misses.append(miss)

        return biases, stds, taus, fprs, nums, misses

    def first_cross_consecutive(self, prob_window, tau):
        mask = prob_window >= tau
        run = np.convolve(mask.astype(int), np.ones(self.K, int), mode="valid")
        idx = np.flatnonzero(run == self.K)
        return idx[0] if idx.size > 0 else None

    @staticmethod
    def onset_score(p, ti_star, eps=1e-6):
        """Unnormalized score proportional to P(t* = t_star | x) from per-frame speech probs p."""
        K = len(p)
        if ti_star < 0 or ti_star >= K:
            assert False, "Error"
        p = np.clip(p, eps, 1 - eps)
        ps = np.prod(1 - p[:ti_star]) * np.prod(p[ti_star:])
        ps = np.clip(ps, eps, 1)
        return ps

    def find_onset(self, onset, ti, tau):
        idx = self.first_cross_consecutive(onset, tau)  # First K consecutive samples
        if idx is None:
            ti_hat, miss = 0, 1
            miss = 1
        else:
            ti_hat, miss = ti - self.start_interval + idx, 0
        bias = ti_hat - ti
        return bias, miss

    def collect_onsets(self, target, prob):
        N = len(target)
        # Collect samples of onsets
        for n in range(self.off_interval, N):
            if np.sum(target[n:n+self.on_interval]) == self.on_interval and np.sum(1-target[n-self.off_interval:n])==self.off_interval:
                ti = n + 1
                prob_window = prob[ti-self.start_interval : ti+self.on_interval]
                ti_window = self.start_interval
                onset = (prob_window, ti_window)
                self.onsets.append(onset)

    def onset_bias_estimator(self, targets, probs):
        if len(self.onsets) == 0:
            return None

        self.biases_th = [[] for _ in self.thresholds]

        # Look for onsets in the sequence
        for tau_idx, tau in enumerate(self.thresholds):
            for onset,ti in self.onsets:
                bias, miss = self.find_onset(onset, ti, tau)
                self.metrics_th[tau_idx, 0] += 1
                if miss == 0:
                    bias += self.onset_error_bias
                    self.metrics_th[tau_idx,1] += bias
                    self.metrics_th[tau_idx,2] += bias**2
                    self.biases_th[tau_idx].append(bias)
                self.metrics_th[tau_idx,3] += miss

        return self.get_onset_metric(targets, probs)

    def get_bias_percentiles(self, tau_idx):
        if len(self.biases_th[tau_idx]) == 0:
            return 0, 0, 0
        return np.percentile(self.biases_th[tau_idx], [50, 90, 95])

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
            if self.streaming:
                y_true = y_true[:, -1]
                y_prob = ma.stream_eval(x_batch)
                self.collect_onsets(y_true.cpu().numpy(), y_prob)
            else:
                y_pred = ma.forward(x_batch)
                y_prob = ma.postprocess(y_pred)

            targets.extend(y_true.cpu().numpy().flatten())
            probs.extend(y_prob.flatten())

        # AUROC
        auroc = auroc_score(targets, probs)
        print("AUROC: {:.3f}".format(auroc))

        # Onset Latency
        onset_estimation = None
        if self.streaming:
            onset_estimation = self.onset_bias_estimator(targets, probs)
        if onset_estimation is not None:
            biases, stds, taus, fprs, nums, misses = onset_estimation
            if self.target_tau:
                idx = np.abs(np.array(taus) - self.target_tau).argmin()
            else:
                idx = np.abs(np.array(fprs) - self.target_fpr).argmin()
            p50, p90, p95 = self.get_bias_percentiles(idx)
            print("Onset Estimation for FPR={:.3f}: bias {:.3f} std: {:.3f} p50: {:.1f} p90: {:.1f} p95: {:.1f} "
                  "for threshold {:.2f} of {} onsets, miss detect {}".format(
                fprs[idx], biases[idx], stds[idx], p50, p90, p95, taus[idx], nums[idx], misses[idx]))

        return auroc, onset_estimation

    def __call__(self, ma: ModelAdapter, da: DatasetAdapter):
        return self.evaluate(ma, da.val_loader, desc="Validation")
