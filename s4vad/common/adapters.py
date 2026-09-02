# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
from abc import ABC, abstractmethod
from typing import Dict
import numpy as np
import torch


class ModelAdapter(ABC, torch.nn.Module):
    def __init__(self): super().__init__()
    @classmethod
    @abstractmethod
    def build_from_config(cls, cfg: Dict): ...
    @abstractmethod
    def infer(self, wav: np.ndarray, input_sr: int = None, plot: bool = True): ...


class DatasetAdapter(ABC):
    @classmethod
    @abstractmethod
    def build_from_config(cls, cfg: Dict): ...


class Trainer(ABC):
    @classmethod
    @abstractmethod
    def build_from_config(cls, cfg: Dict): ...
    @abstractmethod
    def __call__(self, model: ModelAdapter, dataset: DatasetAdapter, cfg: Dict): ...


class Evaluator(ABC):
    @classmethod
    @abstractmethod
    def build_from_config(cls, cfg: Dict): ...
    @abstractmethod
    def __call__(self, model: ModelAdapter, dataset: DatasetAdapter, cfg: Dict): ...
