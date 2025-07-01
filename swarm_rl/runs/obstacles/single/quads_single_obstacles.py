from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.single.single_quad_obstacle_baseline import QUAD_BASELINE_CLI_1

_params = ParamGrid(
    [
        ("seed", [0000, 1111, 2222, 3333]),
        ("quads_num_agents", [1]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_1 + (
    ' --quads_encoder_type=corl --rnn_size=15 --quads_obst_hidden_size=15 '
    '--with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones --wandb_group=single_test'
)

_experiment = Experiment(
    "single-test",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_single", experiments=[_experiment])
