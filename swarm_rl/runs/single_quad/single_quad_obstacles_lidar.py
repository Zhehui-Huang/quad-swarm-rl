import os

from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_restart_behavior = os.environ.get("QUAD_LIDAR_RESTART_BEHAVIOR", "resume")
if _restart_behavior not in ("resume", "overwrite"):
    raise ValueError(
        "QUAD_LIDAR_RESTART_BEHAVIOR must be either 'resume' or 'overwrite', "
        f"got {_restart_behavior!r}"
    )

_stage = os.environ.get("QUAD_LIDAR_STAGE", "stage2")
if _stage not in ("stage1", "stage2"):
    raise ValueError(
        "QUAD_LIDAR_STAGE must be either 'stage1' or 'stage2', "
        f"got {_stage!r}"
    )

_checkpoint_kind = os.environ.get("QUAD_LIDAR_CHECKPOINT_KIND", "latest")
if _checkpoint_kind not in ("latest", "best"):
    raise ValueError(
        "QUAD_LIDAR_CHECKPOINT_KIND must be either 'latest' or 'best', "
        f"got {_checkpoint_kind!r}"
    )

_params = ParamGrid([
    ('seed', [0000]),
])

COMMON_LIDAR_CLI = (
    ' --quads_use_obstacles=True --quads_room_dims 12 12 10 --quads_obst_spawn_area 10 10 '
    '--quads_obstacle_obs_type=lidar --quads_obst_min_clearance=0.8 '
    '--quads_lidar_sector_angle=30.0 --quads_lidar_sector_samples=5 '
    '--quads_use_downwash=False --quads_vel_penalty_limit=1.6 --quads_velocity_yaw_max_speed=1.2 '
    '--quads_collision_reward=8.0 --quads_collision_smooth_max_penalty=6.0 '
    '--quads_progress_reward=0.8 --quads_success_reward=3.0 --quads_first_success_reward=10.0 '
    '--quads_vel_reward=0.4 --quads_orient_reward=2.0 --quads_spin_reward=0.5 '
    '--quads_mode=o_random --quads_obs_repr=xyz_vxyz_R_omega_wall '
    '--quads_control_type=velocity_yaw_body_avoid '
    '--quads_avoid_radius=1.0 --quads_cbf_safe_distance=0.3 --quads_cbf_alpha=2.0 '
    '--quads_avoid_lidar_filter_alpha=0.35 --quads_avoid_activation_hysteresis=0.08 '
    '--quads_avoid_floor_guard_z=1.25 --quads_avoid_floor_guard_kp=1.8 '
    '--quads_avoid_floor_guard_max_vz=0.9 '
    f'--load_checkpoint_kind={_checkpoint_kind} '
)

STAGE1_CLI = QUAD_BASELINE_CLI + COMMON_LIDAR_CLI + (
    '--quads_obst_density=0.04 --quads_obst_size=0.30 --quads_obst_collision_reward=4.0 '
    '--replay_buffer_sample_prob=0.0 --train_for_env_steps=3000000 '
    f'--restart_behavior={_restart_behavior}'
)

STAGE2_CLI = QUAD_BASELINE_CLI + COMMON_LIDAR_CLI + (
    '--quads_obst_density=0.16 --quads_obst_size=0.40 --quads_obst_collision_reward=8.0 '
    '--replay_buffer_sample_prob=0.25 --train_for_env_steps=25000000 '
    f'--restart_behavior={_restart_behavior}'
)

QUAD_OBSTACLE_LIDAR_CLI = dict(stage1=STAGE1_CLI, stage2=STAGE2_CLI)[_stage]

_experiment = Experiment(
    'single_obstacles_lidar_body_cbf_v5',
    QUAD_OBSTACLE_LIDAR_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_quad_obstacles_lidar_body_cbf_v5', experiments=[_experiment])
