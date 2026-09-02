# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
import argparse
from typing import Dict
from hyperpyyaml import load_hyperpyyaml
from s4vad.common.registries import build
from utils import seed_everything


def argument_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", '-y', type=str, required=True)
    parser.add_argument("--seed", '-s', type=int, default=2025)
    parser.add_argument("--model_path", '-p', type=str, default=None)
    args_dict = vars(parser.parse_args())
    return args_dict


def evaluate(cfg: Dict):
    dataset_adapter = build("dataset", cfg["dataset"], cfg["dataset_cfg"])
    model_adaptor = build("model", cfg["model"], cfg["model_cfg"])
    evaluator = build("evaluator", cfg["evaluator"], cfg["evaluator_cfg"])
    evaluator(model_adaptor, dataset_adapter)


if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    args = argument_handler()
    with open(args['yaml_path']) as yaml_stream:
        eval_cfg = dict(load_hyperpyyaml(yaml_stream))
    if args["model_path"] is not None:
        eval_cfg["model_cfg"]["path"] = args["model_path"]

    seed_everything(args["seed"])

    # ----------------------------
    # Eval
    # ----------------------------
    evaluate(eval_cfg)
