# Drone Simulator: QuadSwarm

A codebase for training reinforcement learning policies for quadrotor swarms.
Includes:
* Flight dynamics simulator forked from https://github.com/amolchanov86/gym_art
and extended to support swarms of quadrotor drones
* Scripts and the necessary wrappers to facilitate training of control policies with Sample Factory
https://github.com/alex-petrenko/sample-factory

[//]: # (**Paper:** https://arxiv.org/abs/2109.07735)

[//]: # ()
[//]: # (**Website:** https://sites.google.com/view/swarm-rl)


<p align="middle">

<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Swarm_vs_Swarm.gif?raw=true" width="45%">
</p> 

<p align="middle">
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Obstacles_Static_Same_Goal.gif?raw=true" width="45%">
&emsp;
<img src="https://github.com/Zhehui-Huang/quad-swarm-rl/blob/master/swarm_rl/gifs/Scale_32_Static_Same_Goal.gif?raw=true" width="45%">
</p> 

## Installation

Initialize a Python environment, i.e. with `conda` (Python versions >=3.11 are supported):

```
conda create -n swarm-rl python=3.11
conda activate swarm-rl
```

Clone and install this repo as an editable Pip package:
```
git clone https://github.com/Zhehui-Huang/quad-swarm-rl.git
cd quad-swarm-rl
pip install -e .
```

This should pull and install all the necessary dependencies including PyTorch.

## Running experiments

### Train

This will run the baseline experiment.
Change the number of workers appropriately to match the number of logical CPU cores on your machine, but it is advised that
the total number of simulated environments is close to that in the original command:

We provide a training script `train.sh`, so you can simply start training by command `bash train.sh`.

Or, even better, you can use the runner scripts in `swarm_rl/runs/`. These runner scripts (a Sample Factory feature) are Python files that
contain experiment parameters, and support features such as evaluation on multiple seeds and gridsearches.

To execute a runner script run the following command:

```
python -m sample_factory.launcher.run --run=swarm_rl.runs.single_quad.single_quad --max_parallel=4 --pause_between=1 --experiments_per_gpu=1 --num_gpus=4
```

This command will start training four different seeds in parallel on a 4-GPU server. Adjust the parameters accordingly to match
your hardware setup.

To monitor the experiments, go to the experiment folder, and run the following command:

```
tensorboard --logdir=./
```
### WandB support

If you want to monitor training with WandB, follow the steps below: 
- setup WandB locally by running `wandb login` in the terminal (https://docs.wandb.ai/quickstart#1.-set-up-wandb).
* add `--with_wandb=True` in the command.

Here is a total list of wandb settings: 
```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_project: WandB "Project" (default: sample_factory)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

### Test
To test the trained model, run the following command:

```
python -m swarm_rl.enjoy --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --quads_use_numba=False --quads_render=True --train_dir=PATH_TO_TRAIN_DIR --experiment=EXPERIMENT_NAME --quads_view_mode CAMERA_VIEWS
```
EXPERIMENT_NAME and PATH_TO_TRAIN_DIR can be found in the cfg.json file of your trained model

CAMERA_VIEWS can be any number of views from the following: `[topdown, global, chase, side, corner0, corner1, corner2, corner3, topdownfollow]`


## Unit Tests

To run unit tests:

```
./run_tests.sh
```

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our following papers.

### QuadSwarm: A Modular Multi-Quadrotor Simulator for Deep Reinforcement Learning with Direct Thrust Control 
[ICRA Workshop: The Role of Robotics Simulators for Unmanned Aerial Vehicles, 2023](https://imrclab.github.io/workshop-uav-sims-icra2023/)

Drone Simulator for Reinforcement Learning.
```
@article{huang2023quadswarm,
  title={Quadswarm: A modular multi-quadrotor simulator for deep reinforcement learning with direct thrust control},
  author={Huang, Zhehui and Batra, Sumeet and Chen, Tao and Krupani, Rahul and Kumar, Tushar and Molchanov, Artem and Petrenko, Aleksei and Preiss, James A and Yang, Zhaojing and Sukhatme, Gaurav S},
  journal={arXiv preprint arXiv:2306.09537},
  year={2023}
}
```

### Sim-to-(Multi)-Real: Transfer of Low-Level Robust Control Policies to Multiple Quadrotors
IROS 2019

Single drone: a unified control policy adaptable to various types of physical quadrotors.
```
@inproceedings{molchanov2019sim,
  title={Sim-to-(multi)-real: Transfer of low-level robust control policies to multiple quadrotors},
  author={Molchanov, Artem and Chen, Tao and H{\"o}nig, Wolfgang and Preiss, James A and Ayanian, Nora and Sukhatme, Gaurav S},
  booktitle={2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={59--66},
  year={2019},
  organization={IEEE}
}
```
### Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning
CoRL 2021

Multiple drones: a decentralized control policy for multiple drones in obstacle free environments.
```
@inproceedings{batra21corl,
  author    = {Sumeet Batra and
               Zhehui Huang and
               Aleksei Petrenko and
               Tushar Kumar and
               Artem Molchanov and
               Gaurav S. Sukhatme},
  title     = {Decentralized Control of Quadrotor Swarms with End-to-end Deep Reinforcement Learning},
  booktitle = {5th Conference on Robot Learning, CoRL 2021, 8-11 November 2021, London, England, {UK}},
  series    = {Proceedings of Machine Learning Research},
  publisher = {{PMLR}},
  year      = {2021},
  url       = {https://arxiv.org/abs/2109.07735}
}
```
### Collision Avoidance and Navigation for a Quadrotor Swarm Using End-to-end Deep Reinforcement Learning
ICRA 2024

Multiple drones: a decentralized control policy for multiple drones in obstacle dense environments.
```
@inproceedings{huang2024collision,
  title={Collision avoidance and navigation for a quadrotor swarm using end-to-end deep reinforcement learning},
  author={Huang, Zhehui and Yang, Zhaojing and Krupani, Rahul and {\c{S}}enba{\c{s}}lar, Bask{\i}n and Batra, Sumeet and Sukhatme, Gaurav S},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={300--306},
  year={2024},
  organization={IEEE}
}
```
### HyperPPO: A scalable method for finding small policies for robotic control
ICRA 2024

A method to find the smallest control policy for deployment: train once, get tons of models with different size by using HyperNetworks.

We only need four neurons to control a quadrotor! That is super amazing!

Please check following videos for more details:
- [Square Grid Trajectory](https://www.youtube.com/watch?v=IenGT_TOwGQ&ab_channel=USCRESL)
- [Bezier Curve Trajectory](https://www.youtube.com/watch?v=B5EpKlD5F68&ab_channel=USCRESL)
```
@inproceedings{hegde2024hyperppo,
  title={Hyperppo: A scalable method for finding small policies for robotic control},
  author={Hegde, Shashank and Huang, Zhehui and Sukhatme, Gaurav S},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={10821--10828},
  year={2024},
  organization={IEEE}
}
```

### The next paper will appear soon. Hopefully, it will be the end of 2024. That would be a big surprise. 
### After that paper publish, we will spend more time on the community, including maintain a website to cover more details of using this simulator and open source some important trained models. 

Github issues and pull requests are welcome.
