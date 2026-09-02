# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import os
import shutil
import time
from typing import Dict

from tqdm import tqdm
from datetime import datetime
from s4vad.common.adapters import DatasetAdapter, ModelAdapter
from s4vad.common.registries import register
from s4vad.common.metrics import auroc_score
from utils import get_device
from torch.nn.utils import clip_grad_norm_
import torch



@register("trainer", "vad_trainer")
class VadTrainer(object):
    def __init__(self,
                 epochs=10,
                 evaluate_every_n_epochs=1,
                 gradient_max_norm=5,
                 frame_level=True,
                 loss_func=torch.nn.functional.binary_cross_entropy_with_logits,
                 optimizer=torch.optim.AdamW,
                 optimizer_cfg=None,
                 scheduler=None,
                 scheduler_cfg=None,
                 warmup=False,
                 save_last=False,
                 out_dir=".",
                 **kwargs):
        self.epochs = epochs
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.frame_level = frame_level
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler = scheduler
        self.scheduler_cfg = scheduler_cfg or {}
        self.gradient_max_norm = gradient_max_norm
        self.warmup = warmup
        self.save_last = save_last
        self.model_path = ""
        out_folder = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
        self.out_dir = os.path.join(out_dir, out_folder)
        os.makedirs(self.out_dir, exist_ok=True)
        yaml_path = kwargs.get("yaml_path")
        shutil.copy(yaml_path, self.out_dir)
        self.device = get_device()
        self.losses = []

    @classmethod
    def build_from_config(cls, cfg: Dict):
        return cls(**cfg) if cfg else cls()

    def save_best_model(self, model, epoch):
        # Remove old model
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        # Save new model
        self.model_path = os.path.join(self.out_dir, f'model_epoch{epoch + 1}.pth')
        torch.save(model.state_dict(), self.model_path)

    def save_last_model(self, model):
        self.model_path = os.path.join(self.out_dir, f'model_last.pth')
        torch.save(model.state_dict(), self.model_path)


    def train_step(self, x_batch, y_true, model):
        # Put batch on device
        x_batch = x_batch.to(self.device)
        y_true = y_true.to(self.device)
        # Forward Pass
        y_pred = model(x_batch)
        # Loss
        loss = self.loss_func(y_pred, y_true)
        # Update Loss Metrics
        self.losses.append(loss.item())
        # Compute Gradients
        loss.backward()
        # Clip gradients
        clip_grad_norm_(model.parameters(), self.gradient_max_norm, norm_type=2)
        # Backward Pass
        self.optimizer.step()
        # Zero Out Gradients
        self.optimizer.zero_grad()

    def val_step(self, x_batch, y_true, ma):
        if isinstance(x_batch, list):
            x_batch = torch.concat([item for item in x_batch])
            y_true = torch.concat([item for item in y_true])
        # Put batch on device
        x_batch = x_batch.to(self.device)
        y_true = y_true.to(self.device)
        # Forward Pass
        y_pred = ma.forward(x_batch)
        # Postprocess
        y_prob = ma.postprocess(y_pred)
        return y_prob, y_true if self.frame_level else y_true[:,-1]

    def __call__(self, ma: ModelAdapter, da: DatasetAdapter):
        # Optimizer and scheduler
        self.optimizer = self.optimizer(ma.model.parameters(), **self.optimizer_cfg)
        if self.scheduler:
            self.scheduler = self.scheduler(self.optimizer, **self.scheduler_cfg)

        ma.model.to(self.device)
        ma.model.train()

        # Warm-Up
        if self.warmup:
            print("Warming up...")
            warmup_steps = 100
            base_lr = self.optimizer_cfg.get("lr",1e-3)
            warmup_lr = base_lr*0.1  # 10× smaller
            for pg in self.optimizer.param_groups:
                pg['lr'] = warmup_lr
            for step, (x_batch, y_true) in enumerate(da.train_loader):
                if step >= warmup_steps:
                    break
                # Training step
                self.train_step(x_batch, y_true, ma.model)
            # Set LR back
            for pg in self.optimizer.param_groups:
                pg['lr'] = base_lr

        print("Start training...")
        best_metric = 0
        # Iterate Over Train Dataloader
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            # Iterate over training dataloader
            ma.model.train()
            n_steps = len(da.train_loader)
            start = time.time()
            self.losses.clear()
            for step, (x_batch, y_true) in enumerate(da.train_loader):
                # Training step
                self.train_step(x_batch, y_true, ma.model)
                if step % 200 == 0 or step == n_steps - 1:
                    avg_loss = sum(self.losses) / len(self.losses)
                    print(f"step: {step}/{n_steps} ({int(100*step/n_steps)}%), loss: {avg_loss:.4f}")

            stop = time.time()
            print(f"Epoch {epoch+1} took {(stop-start)/60:.1f} min.")
            if self.scheduler:
                self.scheduler.step()

            # Validation
            if epoch % self.evaluate_every_n_epochs == 0 or epoch == self.epochs - 1:
                # Iterate Over Validation Dataloader
                print(f'Validate epoch {epoch+1}...')
                ma.model.eval()
                targets, probs = [], []
                for step, (x_batch, y_true) in tqdm(enumerate(da.val_loader)):
                    # Validation step
                    y_prob, y_true = self.val_step(x_batch, y_true, ma)
                    targets.extend(y_true.cpu().numpy().flatten())
                    probs.extend(y_prob.flatten())

                auroc = auroc_score(targets, probs)
                print(f"AUROC={auroc:.4f}, best AUROC={best_metric:.4f}")
                if best_metric < auroc:
                    self.save_best_model(ma.model, epoch)
                    print(f"Got AUROC={auroc:.4f} improvement from {best_metric:.4f}. Saving best model to {self.model_path}")
                    best_metric = auroc
                # clear cache
                torch.cuda.empty_cache()

        if self.save_last:
            self.save_last_model(ma.model)
            print(f"Saving last model to {self.model_path}")

        print(f"Training complete!")
