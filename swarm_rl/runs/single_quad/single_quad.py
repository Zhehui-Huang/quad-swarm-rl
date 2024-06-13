from sample_factory.launcher.run_description import RunDescription, Experiment, ParamGrid
from swarm_rl.runs.single_quad.baseline import QUAD_BASELINE_CLI

_params = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333]),
    ("quads_obs_rel_rot", [True]),
])

SINGLE_CLI = QUAD_BASELINE_CLI + (
    ' --with_wandb=True --wandb_project=Quad-Swarm-RL --wandb_group=single_update_model_2 --wandb_user=multi-drones'
)

_experiment = Experiment(
    'single_update_model',
    SINGLE_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('single_drone', experiments=[_experiment])

# Command to use this script on local machine: Please change num_workers to the physical cores of your local machine
# python -m sample_factory.launcher.run --run=swarm_rl.runs.quad_multi_mix_baseline --backend=processes --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
