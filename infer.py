# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import Dict
from hyperpyyaml import load_hyperpyyaml
from s4vad.common.registries import build
from torchaudio import load
import numpy as np
from utils import seed_everything


def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", '-y', type=str, required=True)
    parser.add_argument("--audio_path", '-i', type=str, required=True)
    parser.add_argument("--seed", '-s', type=int, default=2025)
    args_dict = vars(parser.parse_args())
    return args_dict


if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    args = argument_handler()
    with open(args['yaml_path']) as yaml_stream:
        eval_cfg = dict(load_hyperpyyaml(yaml_stream))

    seed_everything(args["seed"])

    # Load wav
    wav, sr = load(args['audio_path'])

    # Init model adaptor
    model_adaptor = build("model", eval_cfg["model"], eval_cfg["model_cfg"])

    # ----------------------------
    # Infer
    # ----------------------------
    model_adaptor.infer(wav, input_sr=sr)
