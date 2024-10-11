from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.multi_drones.quad_multi_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [6666]),
        ("quads_obst_collision_prox_weight", [0.001, 0.01, 0.1, 0.5]),
        ("quads_obst_collision_prox_min", [0.01, 0.05, 0.1]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --quads_room_dims 7.5 7.5 5.0 --quads_obst_spawn_area 7.5 5.0 '
    '--num_envs_per_worker=4 --rnn_size=16 --quads_obs_repr=xyz_vxyz_R_omega --quads_dynamic_goal=True --normalize_input=True '
    ' --replay_buffer_sample_prob=0.3 --quads_obst_grid_size=0.5 --quads_obst_spawn_center=False --quads_obst_grid_size_range 0.5 0.8 --quads_obst_grid_size_random=True '
    '--quads_neighbor_visible_num=2 --quads_neighbor_obs_type=pos_vel --quads_neighbor_hidden_size=16 '
    '--quads_obstacle_tof_resolution=8 --quads_obst_collision_reward=5.0 --quads_obst_collision_prox_max=0.5 '
    '--quads_obst_hidden_size=16 --quads_obst_density=0.2 --quads_obstacle_obs_type=ToFs --quads_mode=o_random_dynamic_goal '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=darren-personal '
    '--wandb_group=md_mo_pmin_sweep_2'
)

_experiment = Experiment(
    "md_mo",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("eight_drone_multi_obst", experiments=[_experiment])