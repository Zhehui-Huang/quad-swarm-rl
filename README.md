# Quad Swarm RL Single

本仓库维护一个单无人机激光雷达避障穿越任务。仿真基于 `gym_art` 四旋翼动力学，训练使用
Sample Factory/APPO。当前保留的主线是 `lidar body CBF`：策略接收机体系 9 方向扇区雷达，
输出机体系速度指令，低层通过 CBF-QP 安全过滤器限制朝障碍物的不安全速度分量。

评估目标不是确认脚本能跑完，而是判断无人机能否持续穿过随机柱状障碍场并到达连续目标点。

## 环境安装

推荐 Python 3.11：

```bash
conda create -n swarm-rl python=3.11
conda activate swarm-rl
pip install -e .
```

可视化和部分测试依赖 OpenGL。无桌面服务器上如需渲染，可用 `xvfb-run` 包裹命令。

## 快速使用

运行单元测试和编译检查：

```bash
bash run_tests.sh
```

从 v5 best checkpoint 创建独立 v6 雷达避障微调实验，并开始训练：

```bash
bash train_single_quad_obstacles_lidar.sh
```

继续已有 v6 实验：

```bash
bash train_single_quad_obstacles_lidar.sh resume-v6
```

确定性评估 v6 latest checkpoint，默认 50 个 episode：

```bash
bash test_single_quad_obstacles_lidar.sh latest v6
```

评估 best checkpoint，并指定 100 个 episode：

```bash
bash test_single_quad_obstacles_lidar.sh best v6 100
```

手动可视化策略时仍可直接使用 `swarm_rl.enjoy`，但正式结果以 `swarm_rl.evaluate` 输出为准。

## 训练入口

主入口：

```bash
bash train_single_quad_obstacles_lidar.sh [mode]
```

常用模式：

| 模式 | 作用 |
| --- | --- |
| `finetune-v6` | 默认模式，从 v5 best 创建 v6 并微调 |
| `resume-v6` | 继续最新 v6 checkpoint |
| `resume` | 继续旧 v5 最新阶段二 checkpoint |
| `retrain` | 覆盖并完整重训 v5 阶段一和阶段二 |
| `stage1` | 只续训 v5 阶段一 |
| `stage1-retrain` | 覆盖并重训 v5 阶段一 |
| `stage2` | 只续训 v5 阶段二 |
| `stage2-retrain` | 不加载旧权重，直接从零训练 v5 阶段二 |
| `stage2-from-v4` | 复制 v4 checkpoint，并用 v5 参数训练阶段二 |
| `two-stage` | 依次续训 v5 阶段一和阶段二 |

v6 也可以直接启动：

```bash
bash train_single_quad_obstacles_lidar_v6.sh bootstrap
bash train_single_quad_obstacles_lidar_v6.sh resume
```

`bootstrap` 会调用 `swarm_rl/bootstrap_lidar_v6.py`，从 v5 best checkpoint 创建新的 v6
实验目录。若 v6 目录已经存在，脚本会改为从 latest checkpoint 继续。

## 评估方法

正式评估入口：

```bash
bash test_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes]
bash eval_single_quad_obstacles_lidar.sh [latest|best] [v6|v5] [episodes] [cpu|gpu]
```

等价 Python 命令示例：

```bash
python -m swarm_rl.evaluate \
  --algo=APPO \
  --env=quadrotor_multi \
  --train_dir=./train_dir \
  --experiment=single_quad_obstacles_lidar_body_cbf_v6/single_obstacles_lidar_body_cbf_v6_/00_single_obstacles_lidar_body_cbf_v6_see_0/ \
  --load_checkpoint_kind=latest \
  --quads_episode_duration=45.0 \
  --eval_num_episodes=50
```

评估默认使用确定性动作、关闭渲染和 replay buffer。episode 不会在首次到达目标点后提前结束；
环境会继续生成下一个目标点。碰撞率和成功率按“到达的目标段”统计，而不是只按整条 episode 统计：

| 指标 | 含义 |
| --- | --- |
| 成功率 | 无碰撞到达目标点的次数 / 到达目标点总次数 |
| 到达率 | 至少到达过一个目标点的 episode 比例 |
| 碰撞率 | 到达目标点前发生过碰撞的目标段数 / 到达目标点总次数 |
| episode 碰撞率 | 整条 episode 内出现过障碍物或房间碰撞的比例 |
| 平均通过时间 | 首次到达目标区域的时间；未到达时按 episode 结束时间记录 |
| 平均目标段时间 | 每次到达目标点前的平均用时 |
| 平均目标段路径长度 | 每次到达目标点前的平均轨迹长度 |
| 最小障碍物距离 | 9 方向雷达含墙面距离的 episode 内最小净空，仅作参考 |
| 轨迹平滑性 | 平均速度变化率，数值越低通常越平滑 |
| 卡死率 | 出现连续低速且目标距离无明显改善窗口的 episode 比例 |

## 成功率判定

默认成功标准位于 `test_single_quad_obstacles_lidar.sh` 和 `swarm_rl/evaluate.py`：

```text
每次到达目标点都会结算一次目标段
该目标段内没有障碍物碰撞、房间边界碰撞或无人机碰撞，则该目标点计为成功
success_rate = 无碰撞目标点数 / 到达目标点总数
collision_rate = 有碰撞目标段数 / 到达目标点总数
```

这种口径不会因为 episode 在首次到达后继续飞行而把后续路径混入第一次通过结果。最小障碍物距离、
整条 episode 路径长度和轨迹平滑性仍会输出，用于诊断策略风格，但不参与成功率判定。

如需做更保守验收，可显式覆写：

```bash
python -m swarm_rl.evaluate ... \
  --eval_success_max_path_ratio=2.5
```

## 输出位置

评估结果写入：

```text
eval_results/lidar_eval_YYYYMMDD_HHMMSS.json
eval_results/lidar_eval_YYYYMMDD_HHMMSS.csv
```

JSON 包含 summary、阈值和逐 episode 明细；CSV 便于画图和与不同 checkpoint 横向比较。

训练结果仍位于：

```text
train_dir/<experiment_name>/
```

## 项目结构

```text
gym_art/quadrotor_multi/              四旋翼动力学、碰撞、障碍物、雷达和渲染
gym_art/quadrotor_multi/obstacles/    柱状障碍物生成、扇区雷达和单元测试
swarm_rl/env_wrappers/                Sample Factory 环境封装、参数和 reward shaping
swarm_rl/models/                      策略网络和注意力层
swarm_rl/runs/single_quad/            单无人机 lidar 实验配置
swarm_rl/bootstrap_lidar_v6.py        从 v5 best 创建 v6 微调 checkpoint
swarm_rl/evaluate.py                  复合指标评估入口
train_single_quad_obstacles_lidar.sh  主训练入口
train_single_quad_obstacles_lidar_v6.sh  v6 微调训练入口
test_single_quad_obstacles_lidar.sh   正式评估入口
TRAINING_IMPROVEMENTS.md              训练、控制和评估改进记录
```

## 常见问题

**best checkpoint 一定比 latest 好吗？**

不一定。best 按训练 reward 保存，reward 会受到惩罚退火和探索噪声影响。模型选型应优先比较评估 JSON 中的目标段成功率、目标段碰撞率、平均目标段时间和轨迹平滑性。

**为什么到达率高但成功率低？**

这通常说明策略能找到目标，但在到达目标点前发生过碰撞。到达率只回答“有没有到”，成功率回答“到达目标点的过程中是否无碰撞”。

**评估时没有画面是不是正常？**

正常。正式评估默认关闭渲染以保证速度和可复现性。需要人工观察时运行 `python -m swarm_rl.enjoy ... --quads_render=True`。

**没有 `train_dir` 怎么办？**

先训练或解压已有训练结果快照。评估脚本会检查 `train_dir/<experiment>/config.json`，缺失时会直接报错。

## 来源

核心仿真代码源自 [gym_art](https://github.com/amolchanov86/gym-art)，训练框架使用
[Sample Factory](https://github.com/alex-petrenko/sample-factory)。本仓库已整理为单无人机 lidar body CBF 避障穿越实验工程。
