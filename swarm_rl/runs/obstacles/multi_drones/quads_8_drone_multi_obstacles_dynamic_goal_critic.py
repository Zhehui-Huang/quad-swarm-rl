from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.multi_drones.quad_multi_obstacle_baseline import QUAD_BASELINE_CLI_8

_params = ParamGrid(
    [
        ("seed", [0000, 3333]),
        ("quads_obst_min_gap_threshold", [0.2, 0.3, 0.4]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # Obst related
    ' --quads_room_dims 8.0 8.0 5.0 --quads_obst_spawn_area 6 4 --quads_obst_grid_size=1.0 '
    '--quads_obst_spawn_center=False --quads_obst_grid_size_random=False --quads_obst_grid_size_range 1.0 1.0 '
    '--quads_obst_collision_prox_weight=0.01 --quads_obst_collision_prox_min=0.01 --quads_obst_collision_prox_max=0.05 '
    '--quads_obst_density=0.2 '
    # Random
    '--quads_obst_density_random=True --quads_obst_density_min=0.2 --quads_obst_density_max=0.4 '
    '--quads_obst_size_random=True --quads_obst_size_min=0.2 --quads_obst_size_max=0.4 '
    '--quads_obst_noise=0.05 '
    # SBC
    '--quads_enable_sbc=False --quads_neighbor_range=2.0 --quads_obst_range=2.0 --quads_sbc_obst_agg=0.2 '
    '--quads_coeff_sbc_acc=0.2 --quads_coeff_sbc_boundary=0.0 '
    # Aux
    '--normalize_input=True --quads_dynamic_goal=True --exploration_loss_coeff=0.001 --quads_coeff_effort=0.05 '
    '--quads_coeff_omega=0.2 --quads_coeff_spin=0.0 --quads_mode=o_random_dynamic_goal '
    '--quads_critic_rnn_size=64 --quads_critic_obs=octomap --replay_buffer_sample_prob=0.75 '
    # Neighbor related
    '--quads_neighbor_obs_update_freq=50 '
    # W & B
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones --wandb_group=search_min_gap_v2'
)

_experiment = Experiment(
    "search_min_gap_v2",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("eight_drone_multi_obst", experiments=[_experiment])