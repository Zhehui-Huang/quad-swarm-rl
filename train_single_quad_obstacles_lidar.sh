#!/bin/bash

# ==============================================================================
# 单无人机模拟激光雷达避障训练脚本
# ==============================================================================

set -e
cd "$(dirname "$0")"

MODE="${1:-finetune-v6}"
case "${MODE}" in
  finetune-v6|--finetune-v6)
    exec bash train_single_quad_obstacles_lidar_v6.sh bootstrap
    ;;
  resume-v6|--resume-v6)
    exec bash train_single_quad_obstacles_lidar_v6.sh resume
    ;;
  finetune-best|--finetune-best)
    exec bash train_single_quad_obstacles_lidar_v6.sh bootstrap
    ;;
  resume|--resume)
    STAGES=("stage2:resume")
    MODE_LABEL="续训阶段二"
    ;;
  retrain|--retrain|restart|--restart|overwrite|--overwrite)
    STAGES=("stage1:overwrite" "stage2:resume")
    MODE_LABEL="两阶段重训"
    ;;
  stage1|--stage1)
    STAGES=("stage1:resume")
    MODE_LABEL="只运行阶段一"
    ;;
  stage1-retrain|--stage1-retrain)
    STAGES=("stage1:overwrite")
    MODE_LABEL="重训阶段一"
    ;;
  stage2|--stage2)
    STAGES=("stage2:resume")
    MODE_LABEL="只运行阶段二"
    ;;
  stage2-retrain|--stage2-retrain)
    STAGES=("stage2:overwrite")
    MODE_LABEL="从零重训阶段二"
    ;;
  stage2-from-v4|--stage2-from-v4)
    STAGES=("stage2:resume")
    MODE_LABEL="从 v4 checkpoint 直接训练阶段二"
    BOOTSTRAP_FROM_V4=1
    ;;
  two-stage|--two-stage)
    STAGES=("stage1:resume" "stage2:resume")
    MODE_LABEL="两阶段续训"
    ;;
  -h|--help|help)
    echo "用法: bash train_single_quad_obstacles_lidar.sh [finetune-v6|resume-v6|resume|retrain|stage1|stage1-retrain|stage2|stage2-retrain|stage2-from-v4|two-stage]"
    echo "  finetune-v6     默认，从 v5 best 创建独立 v6 并微调"
    echo "  resume-v6       继续最新 v6 checkpoint"
    echo "  resume          继续旧 v5 最新阶段二 checkpoint"
    echo "  retrain         阶段一覆盖旧目录重训，然后阶段二从阶段一 checkpoint 续训"
    echo "  stage1          只续训阶段一"
    echo "  stage1-retrain  只覆盖重训阶段一"
    echo "  stage2          只续训阶段二"
    echo "  stage2-retrain  不加载旧权重，直接从零训练阶段二"
    echo "  stage2-from-v4  复制 v4 最新 checkpoint，并用 v5 参数直接训练阶段二"
    echo "  two-stage       阶段一续训完成后，阶段二续训"
    exit 0
    ;;
  *)
    echo "未知模式: ${MODE}"
    echo "用法: bash train_single_quad_obstacles_lidar.sh [finetune-v6|resume-v6|resume|retrain|stage1|stage1-retrain|stage2|stage2-retrain|stage2-from-v4|two-stage]"
    exit 2
    ;;
esac

echo "开始训练单无人机模拟激光雷达避障任务 (${MODE_LABEL})..."

V5_EXPERIMENT_DIR="train_dir/single_quad_obstacles_lidar_body_cbf_v5/single_obstacles_lidar_body_cbf_v5_/00_single_obstacles_lidar_body_cbf_v5_see_0"
V4_EXPERIMENT_DIR="train_dir/single_quad_obstacles_lidar_body_cbf_v4/single_obstacles_lidar_body_cbf_v4_/00_single_obstacles_lidar_body_cbf_v4_see_0"
EXPERIMENT_CONFIG="${V5_EXPERIMENT_DIR}/config.json"

if [ "${BOOTSTRAP_FROM_V4:-0}" = "1" ]; then
    if [ -e "${V5_EXPERIMENT_DIR}" ]; then
        echo "v5 实验目录已经存在，拒绝覆盖：${V5_EXPERIMENT_DIR}"
        echo "如需继续 v5，请运行：bash train_single_quad_obstacles_lidar.sh resume"
        exit 1
    fi
    if [ ! -f "${V4_EXPERIMENT_DIR}/config.json" ] || [ ! -d "${V4_EXPERIMENT_DIR}/checkpoint_p0" ]; then
        echo "找不到可用的 v4 checkpoint：${V4_EXPERIMENT_DIR}"
        exit 1
    fi

    mkdir -p "${V5_EXPERIMENT_DIR}"
    cp "${V4_EXPERIMENT_DIR}/config.json" "${V5_EXPERIMENT_DIR}/config.json"
    cp -a "${V4_EXPERIMENT_DIR}/checkpoint_p0" "${V5_EXPERIMENT_DIR}/checkpoint_p0"
    echo "已将 v4 checkpoint 复制到 v5，原 v4 日志和权重不会被修改。"
fi

for STAGE_SPEC in "${STAGES[@]}"; do
    export QUAD_LIDAR_STAGE="${STAGE_SPEC%%:*}"
    export QUAD_LIDAR_RESTART_BEHAVIOR="${STAGE_SPEC##*:}"
    export QUAD_LIDAR_CHECKPOINT_KIND="${CHECKPOINT_KIND:-latest}"

    if [ "${QUAD_LIDAR_RESTART_BEHAVIOR}" = "resume" ] && [ ! -f "${EXPERIMENT_CONFIG}" ]; then
        echo "找不到 body_cbf_v5 checkpoint，不能直接续训 ${QUAD_LIDAR_STAGE}。"
        echo "可运行 retrain 完整重训，或运行 stage2-from-v4 直接从现有 v4 权重开始二阶段。"
        exit 1
    fi

    echo "启动 ${QUAD_LIDAR_STAGE} (restart_behavior=${QUAD_LIDAR_RESTART_BEHAVIOR})..."
    python -m sample_factory.launcher.run \
        --run=swarm_rl.runs.single_quad.single_quad_obstacles_lidar \
        --max_parallel=1 \
        --pause_between=1 \
        --experiments_per_gpu=1 \
        --num_gpus=1
done

echo "训练启动完成！"
