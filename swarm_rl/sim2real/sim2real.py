import os
import argparse
from distutils.util import strtobool
from pathlib import Path

from attrdict import AttrDict

from swarm_rl.sim2real.generate_single_with_obst import generate_c_model_single_obst
from swarm_rl.sim2real.sim2real_utils import load_sf_model
from swarm_rl.sim2real.generate_single_no_obst import generate_c_model
from swarm_rl.sim2real.generate_multi_no_obst import generate_c_model_multi_deepset
from swarm_rl.sim2real.generate_multi_drone_with_obst import generate_c_model_attention


def torch_to_c_model(args=None):
    parent_model_dir = Path(args.torch_model_dir)
    sub_model_dir_list = [name for name in os.listdir(parent_model_dir) if os.path.isdir(os.path.join(parent_model_dir, name))]
    for i in range(len(sub_model_dir_list)):
        model_dir = parent_model_dir.joinpath(sub_model_dir_list[i])

        # Load a list of model
        models, c_model_names, cfg = load_sf_model(model_dir, args.model_type)
        output_dir = Path(args.output_dir)

        c_base_name, c_extension = os.path.splitext(args.output_model_name)
        output_folder = None
        for model, c_model_name in zip(models, c_model_names):
            final_c_model_name = f'{c_base_name}_{c_model_name}{c_extension}'

            output_folder = output_dir.joinpath(args.model_type, os.path.join(model_dir.parts[1], model_dir.parts[2]))
            output_path = output_folder.joinpath(final_c_model_name)

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            if args.model_type == 'single':
                generate_c_model(model, str(output_path), str(output_folder), testing=args.testing)
            elif args.model_type == 'multi_deepset':
                generate_c_model_multi_deepset(model, str(output_path), str(output_folder), testing=args.testing, cfg=cfg)
            elif args.model_type == 'single_obst':
                generate_c_model_single_obst(model, str(output_path), str(output_folder), testing=args.testing)
            elif args.model_type == 'multi_obst_attn':
                generate_c_model_attention(model, str(output_path), str(output_folder), testing=args.testing)
            else:
                raise NotImplementedError(f'Model type {args.model_type} is not supported')

        print(f'Successfully generated c model at {output_folder}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_model_dir', type=str,
                        default='torch_models/single',
                        help='Path where the policy and config file is stored')
    parser.add_argument('--output_dir', type=str,
                        default='c_models',
                        help='Where you want the c model to be saved')
    parser.add_argument('--output_model_name', type=str,
                        default='network_evaluate.c',
                        help='Name of the generated c model file')
    parser.add_argument('--testing', type=lambda x: bool(strtobool(x)),
                        default=False,
                        help='Whether or not to save the c model in testing mode. Enable this if you want to run the '
                             'unit test to make sure the output of the c model is the same as the pytorch model. Set '
                             'to False if you want to output a c model that will be actually used for sim2real')
    parser.add_argument('--model_type', type=str,
                        default='single',
                        choices=['single', 'single_obst', 'multi_deepset', 'multi_obst_attn'],
                        help='What kind of model we are working with. '
                             'single: single drone, without neighbor encoder, without obstacle encoder.'
                             'single_obst: single drone, without neighbor encoder, with obstacle encoder.'
                             'multi_deepset: multiple drones, neighbor encoder: deepset, obstacle encoder: N/A.'
                             'multi_obst_attention: multiple drones, neighbor encoder: attention, obstacle encoder: N/A'
                        )

    args = parser.parse_args()
    return AttrDict(vars(args))


if __name__ == '__main__':
    # example use case
    # cfg = AttrDict({
    #     'torch_model_dir': 'swarm_rl/sim2real/torch_models/single',
    #     'output_dir': 'swarm_rl/sim2real/c_models',
    #     'output_model_name': 'model.c',
    #     'testing': True,
    #     'model_type': 'single',
    # })
    # torch_to_c_model(cfg)

    configs = parse_args()
    torch_to_c_model(args=configs)
