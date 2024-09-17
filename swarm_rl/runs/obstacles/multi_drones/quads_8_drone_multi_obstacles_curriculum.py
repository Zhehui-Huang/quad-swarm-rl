from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.multi_drones.quad_multi_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_obst_grid_size_random", [True]),
        ("quads_obstacle_tof_resolution", [4, 8]),
        ("normalize_input", [True]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    ' --num_envs_per_worker=4 --rnn_size=12 --quads_obs_repr=xyz_vxyz_R_omega --quads_obs_rel_rot=False '
    '--quads_obst_grid_size=0.5 --quads_obst_spawn_center=False --quads_obst_grid_size_range 0.5 0.8 '
    '--quads_neighbor_visible_num=2 --quads_neighbor_obs_type=pos --quads_neighbor_hidden_size=12 '
    '--quads_obst_hidden_size=12 --quads_obst_density=0.2 --quads_obstacle_obs_type=ToFs --quads_obst_noise=0.05 '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=darren-personal '
    '--wandb_group=md_mo_curriculum '
    '--quads_dynamic_goal=True '
    '--quads_mode=o_random_dynamic_goal_curriculum '
)

_experiment = Experiment(
    "md_mo_curriculum",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("eight_drone_multi_obst", experiments=[_experiment])