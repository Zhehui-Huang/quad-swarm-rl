from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.multi_drones.quad_multi_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # Obst related
    ' --quads_obstacle_obs_type=octomap --quads_room_dims 8.0 8.0 5.0 --quads_obst_spawn_area 8 4 --quads_obst_grid_size=0.5 '
    '--quads_obst_spawn_center=False --quads_obst_grid_size_range 0.6 1.0 --quads_obst_grid_size_random=True '
    '--quads_obst_collision_prox_weight=0.01 --quads_obst_collision_prox_min=0.05 --quads_obst_collision_prox_max=0.5 '
    # Aux
    '--normalize_input=True --quads_dynamic_goal=True --replay_buffer_sample_prob=0.3 '
    '--quads_mode=o_random_dynamic_goal --exploration_loss_coeff=0.003 '
    # W & B
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones --wandb_group=sdf_test'
)

_experiment = Experiment(
    "sdf_test",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("eight_drone_multi_obst", experiments=[_experiment])