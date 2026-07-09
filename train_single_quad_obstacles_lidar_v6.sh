#!/bin/bash

set -e
cd "$(dirname "$0")"

MODE="${1:-bootstrap}"
SOURCE_DIR="train_dir/single_quad_obstacles_lidar_body_cbf_v5/single_obstacles_lidar_body_cbf_v5_/00_single_obstacles_lidar_body_cbf_v5_see_0"
TARGET_DIR="train_dir/single_quad_obstacles_lidar_body_cbf_v6/single_obstacles_lidar_body_cbf_v6_/00_single_obstacles_lidar_body_cbf_v6_see_0"

case "${MODE}" in
  bootstrap|finetune|--bootstrap|--finetune)
    if [ ! -e "${TARGET_DIR}" ]; then
      python -m swarm_rl.bootstrap_lidar_v6 \
        --source-dir="${SOURCE_DIR}" \
        --target-dir="${TARGET_DIR}"
    else
      echo "v6 实验已经存在，将从最新 v6 checkpoint 继续。"
    fi
    ;;
  resume|--resume)
    if [ ! -f "${TARGET_DIR}/config.json" ]; then
      echo "找不到 v6 实验，请先运行：bash train_single_quad_obstacles_lidar_v6.sh bootstrap"
      exit 1
    fi
    ;;
  -h|--help|help)
    echo "用法: bash train_single_quad_obstacles_lidar_v6.sh [bootstrap|resume]"
    echo "  bootstrap  默认，从 v5 best 创建独立 v6 并开始微调"
    echo "  resume     从最新 v6 checkpoint 继续微调"
    exit 0
    ;;
  *)
    echo "未知模式: ${MODE}"
    exit 2
    ;;
esac

export QUAD_LIDAR_V6_RESTART_BEHAVIOR=resume
python -m sample_factory.launcher.run \
  --run=swarm_rl.runs.single_quad.single_quad_obstacles_lidar_v6 \
  --max_parallel=1 \
  --pause_between=1 \
  --experiments_per_gpu=1 \
  --num_gpus=1
