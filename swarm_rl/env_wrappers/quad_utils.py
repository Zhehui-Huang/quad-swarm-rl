import copy

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.model.actor_critic import create_actor_critic

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper
from swarm_rl.env_wrappers.v_value_map import V_ValueMapWrapper


class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps


def make_quadrotor_env_multi(cfg, render_mode=None, **kwargs):
    from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None
    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type='RelativeSampler', noise_ratio=dyn_randomization_ratio, sampler='normal')

    sense_noise = 'default'
    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    rew_coeff = DEFAULT_QUAD_REWARD_SHAPING['quad_rewards']
    use_replay_buffer = cfg.replay_buffer_sample_prob > 0.0
    if cfg.quads_mode == "sim2real_A" or cfg.quads_mode == "sim2real_B":
        sim2real_scenario = cfg.quads_mode[-1]
        quads_mode = "o_random"
        quads_obst_grid_size = 0.5
    else:
        sim2real_scenario = None
        quads_mode = cfg.quads_mode
        quads_obst_grid_size = cfg.quads_obst_grid_size

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr, obs_rel_rot=cfg.quads_obs_rel_rot, dynamic_goal=False,
        # Neighbor
        neighbor_visible_num=cfg.quads_neighbor_visible_num, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area, obst_obs_type=cfg.quads_obstacle_obs_type,
        obst_noise=cfg.quads_obst_noise, grid_size=quads_obst_grid_size,
        obst_tof_resolution=cfg.quads_obstacle_tof_resolution, obst_spawn_center=cfg.quads_obst_spawn_center,
        obst_grid_size_random=cfg.quads_obst_grid_size_random, obst_grid_size_range=cfg.quads_obst_grid_size_range,
        obst_penalty_range=cfg.quads_obst_collision_smooth_penalty_range,

        # Aerodynamics
        use_downwash=cfg.quads_use_downwash,
        # Numba Speed Up
        use_numba=cfg.quads_use_numba,
        # Scenarios
        quads_mode=quads_mode, sim2real_scenario=sim2real_scenario,
        # Room
        room_dims=cfg.quads_room_dims,
        # Replay Buffer
        use_replay_buffer=use_replay_buffer,
        # Rendering
        quads_view_mode=cfg.quads_view_mode, quads_render=cfg.quads_render,
        # Quadrotor Specific (Do Not Change)
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True,
        # Rendering
        render_mode=cfg.quads_render_mode,
    )

    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_obst_smooth_max'] = cfg.quads_obst_collision_smooth_max_penalty

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        reward_shaping['quad_rewards']['quadcol_bin'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = 0.0
        reward_shaping['quad_rewards']['quadcol_bin_obst_smooth_max'] = cfg.quads_obst_collision_smooth_max_penalty
        annealing = [
            AnnealSchedule('quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps),
            AnnealSchedule('quadcol_bin_obst_smooth_max', cfg.quads_obst_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    with_pbt=cfg.with_pbt)
    env = QuadEnvCompatibility(env)

    if cfg.visualize_v_value:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        actor_critic.eval()

        device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index
        name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
        checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
        checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict["model"])
        env = V_ValueMapWrapper(env, actor_critic, render_mode='rgb_array')

    return env


def make_quadrotor_env(env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    if env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, render_mode, **kwargs)
    else:
        raise NotImplementedError
