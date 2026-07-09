import argparse
import copy
import json
import math
from pathlib import Path

import torch


V6_CONFIG_OVERRIDES = {
    "experiment": "00_single_obstacles_lidar_body_cbf_v6_see_0",
    "train_dir": "./train_dir/single_quad_obstacles_lidar_body_cbf_v6/single_obstacles_lidar_body_cbf_v6_",
    "restart_behavior": "resume",
    "load_checkpoint_kind": "latest",
    "learning_rate": 3e-5,
    "curr_lr": 3e-5,
    "kl_loss_coeff": 0.05,
    "initial_stddev": 0.30,
    "continuous_tanh_scale": 1.2,
    "train_for_env_steps": 5_000_000,
    "anneal_collision_steps": 5_000_000.0,
    "anneal_collision_initial_ratio": 0.125,
    "quads_avoid_lidar_filter_alpha": 0.5,
    "quads_avoid_activation_hysteresis": 0.04,
}


def build_finetune_checkpoint(source_checkpoint, action_stddev=0.30, learning_rate=3e-5):
    checkpoint = copy.deepcopy(source_checkpoint)
    stddev_keys = [key for key in checkpoint["model"] if key.endswith("learned_stddev")]
    if not stddev_keys:
        raise ValueError("source checkpoint does not contain a learned_stddev parameter")

    log_stddev = math.log(action_stddev)
    for key in stddev_keys:
        checkpoint["model"][key].fill_(log_stddev)

    checkpoint["optimizer"]["state"] = {}
    for param_group in checkpoint["optimizer"]["param_groups"]:
        param_group["lr"] = learning_rate

    checkpoint["train_step"] = 0
    checkpoint["env_steps"] = 0
    checkpoint["best_performance"] = -1e9
    checkpoint["curr_lr"] = learning_rate
    return checkpoint


def bootstrap_v6(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    source_checkpoints = sorted((source_dir / "checkpoint_p0").glob("best_*.pth"))
    if not source_checkpoints:
        raise FileNotFoundError(f"no best checkpoint found under {source_dir / 'checkpoint_p0'}")
    source_config = source_dir / "config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"missing source config: {source_config}")
    if target_dir.exists():
        raise FileExistsError(f"target v6 experiment already exists: {target_dir}")

    checkpoint = torch.load(source_checkpoints[-1], map_location="cpu", weights_only=False)
    checkpoint = build_finetune_checkpoint(checkpoint)

    with source_config.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    config.update(V6_CONFIG_OVERRIDES)

    checkpoint_dir = target_dir / "checkpoint_p0"
    checkpoint_dir.mkdir(parents=True)
    with (target_dir / "config.json").open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4, sort_keys=True)
        config_file.write("\n")
    target_checkpoint = checkpoint_dir / "checkpoint_000000000_0.pth"
    torch.save(checkpoint, target_checkpoint)
    return source_checkpoints[-1], target_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Bootstrap lidar body CBF v6 from the v5 best policy.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--target-dir", required=True)
    args = parser.parse_args()

    source, target = bootstrap_v6(args.source_dir, args.target_dir)
    print(f"Bootstrapped v6 from {source}")
    print(f"Created {target}")


if __name__ == "__main__":
    main()
