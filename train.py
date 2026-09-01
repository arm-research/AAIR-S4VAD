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


def train(cfg: Dict, yaml_path: str):
    dataset_adapter = build("dataset", cfg["dataset"], cfg["dataset_cfg"])
    model_adapter = build("model", cfg["model"], cfg["model_cfg"])
    cfg["trainer_cfg"].update({"yaml_path": yaml_path, "cfg": cfg})
    trainer = build("trainer", cfg["trainer"], cfg["trainer_cfg"])
    trainer(model_adapter, dataset_adapter)


if __name__ == "__main__":
    # ----------------------------
    # Parse arguments
    # ----------------------------
    args = argument_handler()
    yaml_path = args['yaml_path']
    with open(yaml_path) as yaml_stream:
        train_cfg = dict(load_hyperpyyaml(yaml_stream))
    if args["model_path"] is not None:
        if not args["model_path"].endswith(".pth"):
            raise ValueError("Training from checkpoint requires a .pth file.")
        train_cfg["model_cfg"]["path"] = args["model_path"]

    seed_everything(args['seed'])

    # ----------------------------
    # Train
    # ----------------------------
    train(train_cfg, yaml_path)
