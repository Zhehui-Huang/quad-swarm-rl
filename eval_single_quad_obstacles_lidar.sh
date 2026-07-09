#!/bin/bash

# ==============================================================================
# 单无人机模拟激光雷达避障评估脚本
# 默认评估 v6 latest checkpoint，可通过参数切换 checkpoint/版本/episode 数/设备。
# ==============================================================================

set -e
cd "$(dirname "$0")"

CHECKPOINT_KIND="${1:-latest}"
EXPERIMENT_VERSION="${2:-v6}"
NUM_EPISODES="${3:-50}"
DEVICE="${4:-cpu}"

case "${DEVICE}" in
  cpu)
    EVAL_DEVICE="cpu"
    ;;
  gpu|cuda)
    EVAL_DEVICE="gpu"
    ;;
  *)
    echo "未知设备类型: ${DEVICE}"
    echo "用法: bash eval_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes] [cpu|gpu]"
    exit 2
    ;;
esac

case "${CHECKPOINT_KIND}" in
  latest|best)
    ;;
  -h|--help|help)
    echo "用法: bash eval_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes] [cpu|gpu]"
    echo "  latest   默认，评估最新 checkpoint"
    echo "  best     评估 reward 最好的 checkpoint"
    echo "  v6       默认，评估独立微调实验"
    echo "  v5       评估旧 v5 实验"
    echo "  episodes 默认 50，确定性评估 episode 数"
    echo "  cpu      默认，使用 CPU 评估"
    echo "  gpu      使用 CUDA/GPU 评估"
    exit 0
    ;;
  *)
    echo "未知 checkpoint 类型: ${CHECKPOINT_KIND}"
    echo "用法: bash eval_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes] [cpu|gpu]"
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
    echo "用法: bash eval_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes] [cpu|gpu]"
    exit 2
    ;;
esac

if [ ! -f "train_dir/${EXPERIMENT_PATH}/config.json" ]; then
  echo "找不到实验目录：train_dir/${EXPERIMENT_PATH}"
  echo "请先运行 bash train_single_quad_obstacles_lidar.sh 完成训练，或者检查 train_dir。"
  exit 1
fi

echo "开始评估：checkpoint=${CHECKPOINT_KIND}, version=${EXPERIMENT_VERSION}, episodes=${NUM_EPISODES}, device=${EVAL_DEVICE}"

conda run -n swarm-rl python -m swarm_rl.evaluate \
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
  --eval_success_max_path_length_m=18.0 \
  --device="${EVAL_DEVICE}"

echo "评估结束。"
