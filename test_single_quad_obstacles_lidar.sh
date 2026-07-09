#!/bin/bash

# ==============================================================================
# 单无人机模拟激光雷达避障测试脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

echo "开始评估单无人机模拟激光雷达避障任务..."

CHECKPOINT_KIND="${1:-latest}"
EXPERIMENT_VERSION="${2:-v6}"
NUM_EPISODES="${3:-50}"

case "${CHECKPOINT_KIND}" in
  latest|best)
    ;;
  -h|--help|help)
    echo "用法: bash test_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes]"
    echo "  latest  默认，测试最新 checkpoint"
    echo "  best    测试 reward 最好的 checkpoint"
    echo "  v6      默认，测试独立微调实验"
    echo "  v5      测试旧 v5 实验"
    echo "  episodes 默认 50，确定性评估 episode 数"
    exit 0
    ;;
  *)
    echo "未知 checkpoint 类型: ${CHECKPOINT_KIND}"
    echo "用法: bash test_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes]"
    exit 2
    ;;
esac

case "${EXPERIMENT_VERSION}" in
  v6)
    EXPERIMENT_PATH="single_quad_obstacles_lidar_body_cbf_v6/single_obstacles_lidar_body_cbf_v6_/00_single_obstacles_lidar_body_cbf_v6_see_0/"
    ;;
  v5)
    EXPERIMENT_PATH="single_quad_obstacles_lidar_body_cbf_v5/single_obstacles_lidar_body_cbf_v5_/00_single_obstacles_lidar_body_cbf_v5_see_0/"
    ;;
  *)
    echo "未知实验版本: ${EXPERIMENT_VERSION}"
    echo "用法: bash test_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes]"
    exit 2
    ;;
esac

if [ ! -f "train_dir/${EXPERIMENT_PATH}/config.json" ]; then
  echo "找不到实验目录：train_dir/${EXPERIMENT_PATH}"
  echo "请先运行 bash train_single_quad_obstacles_lidar.sh 完成训练，或者检查 train_dir。"
  exit 1
fi

python -m swarm_rl.evaluate \
  --algo=APPO \
  --env=quadrotor_multi \
  --replay_buffer_sample_prob=0 \
  --quads_use_numba=False \
  --quads_render=False \
  --quads_episode_duration=45.0 \
  --visualize_projection_map=False \
  --visualize_obstacle_point_cloud=False \
  --quads_vel_penalty_limit=1.6 \
  --quads_velocity_yaw_max_speed=1.2 \
  --train_dir=./train_dir \
  --experiment="${EXPERIMENT_PATH}" \
  --quads_view_mode=chase \
  --load_checkpoint_kind="${CHECKPOINT_KIND}" \
  --eval_num_episodes="${NUM_EPISODES}" \
  --eval_output_dir=./eval_results \
  --eval_success_time_limit_s=45.0 \
  --eval_success_max_path_ratio=3.0 \
  --eval_success_max_path_length_m=18.0

echo "评估结束！"
