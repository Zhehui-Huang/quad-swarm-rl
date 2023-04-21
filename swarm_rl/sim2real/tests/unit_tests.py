import torch
import torch.nn as nn
import argparse
import subprocess
import numpy as np
import os

from pathlib import Path
from swarm_rl.sim2real.sim2real import load_sf_model


def test_compare_torch_to_c_model_outputs_single_drone():
    # set this to whatever your project path is
    project_root = Path.home().joinpath('quad-swarm-rl')
    os.chdir(str(project_root))
    # SF torch model used to generate the c model. Set this to be the dir where you store the torch model
    # you used to generate the c model
    torch_model_dir = 'swarm_rl/sim2real/torch_models/single'
    model = load_sf_model(Path(torch_model_dir))

    # get the pytorch model outputs on a random input observation. You can also set this to be some custom observation
    # if you want to debug a specific observation input
    obs = torch.randn((1, 18))
    obs_dict = {'obs': obs}
    torch_model_out = model.action_parameterization(model.actor_encoder(obs_dict))[1].means.detach().numpy()

    # get the c model outputs on the same observation

    # set this to be the directory where you store the c model generated by the torch model
    c_model_dir = Path('swarm_rl/sim2real/c_models/single')
    c_model_path = c_model_dir.joinpath('model.c')
    shared_lib_path = c_model_dir.joinpath('single.so')
    subprocess.run(
        ['g++', '-fPIC', '-shared', '-o', str(shared_lib_path), str(c_model_path)],
        check=True,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )

    import ctypes
    from numpy.ctypeslib import ndpointer
    lib = ctypes.cdll.LoadLibrary(str(shared_lib_path))
    func = lib.main
    func.restype = None
    func.argtypes = [
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
        ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")
    ]

    indata = obs.flatten().detach().numpy()
    outdata = np.zeros(4).astype(np.float32)
    func(indata, indata.size, outdata)

    assert np.allclose(torch_model_out, outdata)


if __name__ == '__main__':
    test_compare_torch_to_c_model_outputs_single_drone()