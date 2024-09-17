import json
import os
from pathlib import Path

import torch
from attrdict import AttrDict
from sample_factory.model.actor_critic import create_actor_critic
from gymnasium import spaces

from swarm_rl.env_wrappers.quad_utils import make_quadrotor_env_multi
from swarm_rl.train import register_swarm_components


def load_sf_model(model_dir, model_type):
    """
        Load a trained SF pytorch model
    """
    assert model_dir.exists(), f'Path {str(model_dir)} is not a valid path'
    # Load hyper-parameters
    cfg_path = model_dir.joinpath('config.json')
    # Compatibility with sf2
    if not cfg_path.exists():
        cfg_path = model_dir.joinpath('cfg.json')

    with open(cfg_path, 'r') as f:
        args = json.load(f)
    args = AttrDict(args)

    # Manually set some values
    args.visualize_v_value = False
    args.quads_encoder_type = 'attention' if (model_type == 'attention' or model_type == 'multi_obst_attn') else 'corl'
    args.quads_sim2real = True
    args.quads_domain_random = False
    args.quads_obst_density_random = False
    args.quads_obst_density_min = 0
    args.quads_obst_density_max = 0
    args.quads_obst_size_random = False
    args.quads_obst_size_min = 0
    args.quads_obst_size_max = 0
    args.quads_obs_rel_rot = False
    args.quads_obst_noise = 0.0
    args.quads_obst_grid_size = 1.0
    args.quads_render_mode = 'human'

    # Load model
    register_swarm_components()
    # Spawn a dummy env, so we can get the obs and action space info
    env = make_quadrotor_env_multi(args)
    # Get checkpoints under checkpoint_p0
    checkpoint_dir = Path(os.path.join(model_dir, 'checkpoint_p0'))
    model_paths_1 = list(checkpoint_dir.glob('*.pth'))

    # Get checkpoints under checkpoint_p0/milestones
    milestone_dir = Path(os.path.join(model_dir, 'checkpoint_p0/milestones'))
    model_paths_2 = list(milestone_dir.glob('*.pth'))

    model_paths = model_paths_1 + model_paths_2

    models = []
    c_model_names = []
    for model_path in model_paths:
        obs_space = spaces.Dict({
            'obs': env.observation_space,
        })
        model = create_actor_critic(args, obs_space, env.action_space)
        model.load_state_dict(torch.load(model_path)['model'])
        models.append(model)

        # Extract the step number from the model path
        split_path = model_path.stem.split('_')
        prex_name = split_path[0]
        train_step = split_path[-1]

        if 'best' in prex_name:
            c_model_name = f'best_step_{train_step}'
        else:
            c_model_name = f'step_{train_step}'

        c_model_names.append(c_model_name)

    return models, c_model_names, args


def process_layer(name, param, layer_type):
    """
    Convert a torch parameter from the NN into a c-equivalent represented as a string
    """
    if layer_type == 'weight':
        weight = 'static const float ' + name + '[' + str(param.shape[0]) + '][' + str(param.shape[1]) + '] = {'
        for row in param:
            weight += '{'
            for num in row:
                weight += str(num.item()) + ','
            # get rid of comma after the last number
            weight = weight[:-1]
            weight += '},'
        # get rid of comma after the last curly bracket
        weight = weight[:-1]
        weight += '};\n'
        return weight
    else:
        bias = 'static const float ' + name + '[' + str(param.shape[0]) + '] = {'
        for num in param:
            bias += str(num.item()) + ','
        # get rid of comma after last number
        bias = bias[:-1]
        bias += '};\n'
        return bias
