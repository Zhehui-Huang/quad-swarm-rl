import csv
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from sample_factory.enjoy import (
    AttrDict,
    ExperimentStatus,
    Learner,
    argmax_actions,
    create_actor_critic,
    extract_env_info,
    get_rnn_size,
    load_from_checkpoint,
    make_dones,
    make_env_func_batched,
    prepare_and_normalize_obs,
    preprocess_actions,
    render_frame,
    unsqueeze_tensor,
)

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


def _to_python(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _episode_stats(info):
    stats = info.get("episode_extra_stats", {})
    return {key: _to_python(value) for key, value in stats.items()}


def _is_success(stats, cfg):
    targets_reached = stats.get("eval/targets_reached", 0.0)
    target_collisions = stats.get("eval/target_collisions", 0.0)
    reached = targets_reached > 0.0
    obstacle_collisions = stats.get("num_collisions_obst_quad", 0.0) <= 0.0
    room_collisions = stats.get("eval/room_collision_count", 0.0) <= 0.0
    flight_time = stats.get("eval/goal_reach_time_s", stats.get("eval/flight_time_s", np.inf))
    path_length = stats.get("eval/path_length_m", np.inf)
    path_ratio = stats.get("eval/path_efficiency_ratio", np.inf)

    checks = {
        "reached_goal": bool(reached),
        "has_collision_free_target": bool(targets_reached > target_collisions),
        "no_episode_obstacle_collision": bool(obstacle_collisions),
        "no_episode_room_collision": bool(room_collisions),
        "within_time_limit": bool(flight_time <= cfg.eval_success_time_limit_s),
        "path_ratio_ok": bool(path_ratio <= cfg.eval_success_max_path_ratio),
        "path_length_ok": bool(path_length <= cfg.eval_success_max_path_length_m),
    }
    return checks["reached_goal"] and checks["has_collision_free_target"], checks


def _mean(records, key):
    values = [row[key] for row in records if key in row and row[key] is not None and np.isfinite(row[key])]
    return float(np.mean(values)) if values else None


def _rate(records, key):
    if not records:
        return 0.0
    return float(np.mean([1.0 if row.get(key, False) else 0.0 for row in records]))


def _target_collision_rate(records):
    targets = sum(row.get("targets_reached", 0.0) for row in records)
    if targets <= 0.0:
        return 0.0
    collisions = sum(row.get("target_collisions", 0.0) for row in records)
    return float(collisions / targets)


def _summarize(records, cfg):
    targets_reached = sum(row.get("targets_reached", 0.0) for row in records)
    target_collisions = sum(row.get("target_collisions", 0.0) for row in records)
    target_collision_rate = _target_collision_rate(records)
    return {
        "episodes": len(records),
        "targets_reached": float(targets_reached),
        "target_collisions": float(target_collisions),
        "success_rate": float(1.0 - target_collision_rate) if targets_reached > 0.0 else 0.0,
        "arrival_rate": _rate(records, "reached_goal"),
        "collision_rate": target_collision_rate,
        "episode_collision_rate": _rate(records, "collision"),
        "stuck_rate": _rate(records, "stuck"),
        "detour_rate": _rate(records, "detour"),
        "avg_goal_reach_time_s": _mean(records, "goal_reach_time_s"),
        "avg_target_time_s": _mean(records, "avg_target_time_s"),
        "avg_flight_time_s": _mean(records, "flight_time_s"),
        "avg_path_length_m": _mean(records, "path_length_m"),
        "avg_target_path_length_m": _mean(records, "avg_target_path_length_m"),
        "avg_path_efficiency_ratio": _mean(records, "path_efficiency_ratio"),
        "avg_min_obstacle_distance_m": _mean(records, "min_obstacle_distance_m"),
        "avg_mean_accel_change_mps2": _mean(records, "mean_accel_change_mps2"),
        "thresholds": {
            "time_limit_s": cfg.eval_success_time_limit_s,
            "max_path_ratio": cfg.eval_success_max_path_ratio,
            "max_path_length_m": cfg.eval_success_max_path_length_m,
        },
    }


def _write_outputs(records, summary, cfg):
    os.makedirs(cfg.eval_output_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = os.path.join(cfg.eval_output_dir, f"lidar_eval_{stamp}")
    json_path = f"{prefix}.json"
    csv_path = f"{prefix}.csv"

    payload = {
        "summary": summary,
        "episodes": records,
        "experiment": cfg.experiment,
        "checkpoint": cfg.load_checkpoint_kind,
    }
    with open(json_path, "w", encoding="utf-8") as fobj:
        json.dump(payload, fobj, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in records for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return json_path, csv_path


def _print_terminal_summary(summary, json_path, csv_path):
    print("\n评估摘要")
    print(f"  episodes: {summary['episodes']}")
    print(f"  targets_reached: {summary['targets_reached']:.0f}")
    print(f"  target_collisions: {summary['target_collisions']:.0f}")
    print(f"  success_rate: {summary['success_rate']:.2%}")
    print(f"  collision_rate: {summary['collision_rate']:.2%}")
    print(f"  episode_collision_rate: {summary['episode_collision_rate']:.2%}")
    print(f"  arrival_rate: {summary['arrival_rate']:.2%}")
    print(f"  stuck_rate: {summary['stuck_rate']:.2%}")
    print(f"  avg_target_time_s: {summary['avg_target_time_s']:.3f}")
    print(f"  avg_target_path_length_m: {summary['avg_target_path_length_m']:.3f}")
    print(f"  avg_mean_accel_change_mps2: {summary['avg_mean_accel_change_mps2']:.3f}")
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")


def evaluate(cfg):
    cfg = load_from_checkpoint(cfg)
    cfg.num_envs = 1
    cfg.quads_render = False
    cfg.no_render = True
    cfg.eval_deterministic = True
    cfg.max_num_episodes = cfg.eval_num_episodes

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
    )
    env_info = extract_env_info(env, cfg)

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    records = []
    num_frames = 0
    last_render_start = time.time()

    with torch.no_grad():
        while len(records) < cfg.eval_num_episodes:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs["actions"]
            action_distribution = actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]

            render_frame(cfg, env, [], len(records), last_render_start)
            obs, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated)
            infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

            if episode_reward is None:
                episode_reward = rew.float().clone()
            else:
                episode_reward += rew.float()

            num_frames += 1
            for agent_i, done_flag in enumerate(dones.cpu().numpy()):
                if not done_flag:
                    continue

                stats = _episode_stats(infos[agent_i])
                success, checks = _is_success(stats, cfg)
                collision = (
                    stats.get("num_collisions_obst_quad", 0.0) > 0.0
                    or stats.get("eval/room_collision_count", 0.0) > 0.0
                )
                path_ratio = stats.get("eval/path_efficiency_ratio", np.inf)
                min_clearance = stats.get("eval/min_obstacle_distance_m", np.inf)
                stuck = stats.get("eval/stuck_windows", 0.0) > 0.0
                detour = path_ratio > cfg.eval_success_max_path_ratio
                targets_reached = stats.get("eval/targets_reached", 0.0)
                target_collisions = stats.get("eval/target_collisions", 0.0)

                record = {
                    "episode": len(records) + 1,
                    "reward": float(episode_reward[agent_i].item()),
                    "success": success,
                    "reached_goal": stats.get("eval/reached_goal", 0.0) >= 1.0,
                    "targets_reached": targets_reached,
                    "target_collisions": target_collisions,
                    "target_collision_rate": stats.get("eval/target_collision_rate"),
                    "collision": collision,
                    "stuck": bool(stuck),
                    "detour": bool(detour),
                    "goal_reach_time_s": stats.get("eval/goal_reach_time_s"),
                    "avg_target_time_s": stats.get("eval/avg_target_time_s"),
                    "flight_time_s": stats.get("eval/flight_time_s"),
                    "path_length_m": stats.get("eval/path_length_m"),
                    "avg_target_path_length_m": stats.get("eval/avg_target_path_length_m"),
                    "path_efficiency_ratio": path_ratio,
                    "min_obstacle_distance_m": min_clearance,
                    "mean_accel_change_mps2": stats.get("eval/mean_accel_change_mps2"),
                    "near_obstacle_time_s": stats.get("eval/near_obstacle_time_s"),
                    "final_distance_to_goal": stats.get("eval/final_distance_to_goal"),
                }
                record.update({f"check/{key}": value for key, value in checks.items()})
                records.append(record)
                rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
                episode_reward[agent_i] = 0

                if len(records) >= cfg.eval_num_episodes:
                    break

    env.close()
    summary = _summarize(records, cfg)
    json_path, csv_path = _write_outputs(records, summary, cfg)
    _print_terminal_summary(summary, json_path, csv_path)
    return ExperimentStatus.SUCCESS, summary["success_rate"]


def main():
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=True)
    status, _ = evaluate(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
