import copy

import gymnasium as gym
import numpy as np

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from swarm_rl.env_wrappers.compatibility import QuadEnvCompatibility
from swarm_rl.env_wrappers.projection_map import ProjectionMapWrapper
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING, QuadsRewardShapingWrapper


class AnnealSchedule:
    def __init__(self, coeff_name, final_value, anneal_env_steps, initial_value=0.0):
        self.coeff_name = coeff_name
        self.final_value = final_value
        self.anneal_env_steps = anneal_env_steps
        self.initial_value = initial_value

    def value_at(self, env_steps):
        progress = min(max(float(env_steps) / self.anneal_env_steps, 0.0), 1.0)
        return self.initial_value + progress * (self.final_value - self.initial_value)


class BoundedActionWrapper(gym.Wrapper):
    """Keep sampled continuous actions inside the environment action space."""

    def step(self, action):
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(clipped_action)


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

    env = QuadrotorEnvMulti(
        num_agents=cfg.quads_num_agents, ep_time=cfg.quads_episode_duration, rew_coeff=rew_coeff,
        obs_repr=cfg.quads_obs_repr,
        # Neighbor
        neighbor_visible_num=cfg.quads_neighbor_visible_num, neighbor_obs_type=cfg.quads_neighbor_obs_type,
        collision_hitbox_radius=cfg.quads_collision_hitbox_radius,
        collision_falloff_radius=cfg.quads_collision_falloff_radius,
        # Obstacle
        use_obstacles=cfg.quads_use_obstacles, obst_density=cfg.quads_obst_density, obst_size=cfg.quads_obst_size,
        obst_spawn_area=cfg.quads_obst_spawn_area,
        obstacle_scan_resolution=cfg.quads_obstacle_scan_resolution,
        obstacle_obs_type=cfg.quads_obstacle_obs_type,
        lidar_sector_angle=cfg.quads_lidar_sector_angle,
        lidar_sector_samples=cfg.quads_lidar_sector_samples,
        obst_min_clearance=cfg.quads_obst_min_clearance,

        # Aerodynamics
        use_downwash=cfg.quads_use_downwash,
        # Numba Speed Up
        use_numba=cfg.quads_use_numba,
        # Scenarios
        quads_mode=cfg.quads_mode,
        # Room
        room_dims=cfg.quads_room_dims,
        # Replay Buffer
        use_replay_buffer=use_replay_buffer,
        # Rendering
        quads_view_mode=cfg.quads_view_mode, quads_render=cfg.quads_render,
        # Quadrotor Specific (Do Not Change)
        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=False, control_type=cfg.quads_control_type,
        velocity_yaw_max_speed=cfg.quads_velocity_yaw_max_speed,
        avoid_radius=cfg.quads_avoid_radius,
        cbf_safe_distance=cfg.quads_cbf_safe_distance,
        cbf_alpha=cfg.quads_cbf_alpha,
        avoid_lidar_filter_alpha=cfg.quads_avoid_lidar_filter_alpha,
        avoid_activation_hysteresis=cfg.quads_avoid_activation_hysteresis,
        avoid_floor_guard_z=cfg.quads_avoid_floor_guard_z,
        avoid_floor_guard_kp=cfg.quads_avoid_floor_guard_kp,
        avoid_floor_guard_max_vz=cfg.quads_avoid_floor_guard_max_vz,
        # Rendering
        render_mode=render_mode,
    )

    if use_replay_buffer:
        env = ExperienceReplayWrapper(env, cfg.replay_buffer_sample_prob, cfg.quads_obst_density, cfg.quads_obst_size,
                                      cfg.quads_domain_random, cfg.quads_obst_density_random, cfg.quads_obst_size_random,
                                      cfg.quads_obst_density_min, cfg.quads_obst_density_max, cfg.quads_obst_size_min, cfg.quads_obst_size_max)

    reward_shaping = copy.deepcopy(DEFAULT_QUAD_REWARD_SHAPING)

    reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward
    reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = cfg.quads_collision_smooth_max_penalty
    reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward
    reward_shaping['quad_rewards']['orient'] = cfg.quads_orient_reward
    reward_shaping['quad_rewards']['spin'] = cfg.quads_spin_reward
    reward_shaping['quad_rewards']['vel'] = cfg.quads_vel_reward
    reward_shaping['quad_rewards']['vel_limit'] = cfg.quads_vel_penalty_limit
    reward_shaping['quad_rewards']['progress'] = cfg.quads_progress_reward
    reward_shaping['quad_rewards']['success'] = cfg.quads_success_reward
    reward_shaping['quad_rewards']['first_success'] = cfg.quads_first_success_reward
    reward_shaping['quad_rewards']['z'] = cfg.quads_z_reward
    reward_shaping['quad_rewards']['stable_z'] = cfg.quads_stable_z_reward
    reward_shaping['quad_rewards']['stable_spin'] = cfg.quads_stable_spin_reward
    reward_shaping['quad_rewards']['floor_stall'] = cfg.quads_floor_stall_reward
    reward_shaping['quad_rewards']['room_floor'] = cfg.quads_room_floor_reward
    reward_shaping['quad_rewards']['room_wall'] = cfg.quads_room_wall_reward
    reward_shaping['quad_rewards']['room_ceiling'] = cfg.quads_room_ceiling_reward

    # this is annealed by the reward shaping wrapper
    if cfg.anneal_collision_steps > 0:
        initial_ratio = cfg.anneal_collision_initial_ratio
        if not 0.0 <= initial_ratio <= 1.0:
            raise ValueError("anneal_collision_initial_ratio must be in [0, 1]")
        reward_shaping['quad_rewards']['quadcol_bin'] = cfg.quads_collision_reward * initial_ratio
        reward_shaping['quad_rewards']['quadcol_bin_smooth_max'] = \
            cfg.quads_collision_smooth_max_penalty * initial_ratio
        reward_shaping['quad_rewards']['quadcol_bin_obst'] = cfg.quads_obst_collision_reward * initial_ratio
        reward_shaping['quad_rewards']['orient'] = cfg.quads_orient_reward
        reward_shaping['quad_rewards']['spin'] = cfg.quads_spin_reward
        reward_shaping['quad_rewards']['vel'] = cfg.quads_vel_reward * initial_ratio
        reward_shaping['quad_rewards']['vel_limit'] = cfg.quads_vel_penalty_limit
        reward_shaping['quad_rewards']['progress'] = cfg.quads_progress_reward
        reward_shaping['quad_rewards']['success'] = cfg.quads_success_reward
        reward_shaping['quad_rewards']['first_success'] = cfg.quads_first_success_reward
        reward_shaping['quad_rewards']['z'] = cfg.quads_z_reward
        reward_shaping['quad_rewards']['stable_z'] = cfg.quads_stable_z_reward
        reward_shaping['quad_rewards']['stable_spin'] = cfg.quads_stable_spin_reward
        reward_shaping['quad_rewards']['floor_stall'] = cfg.quads_floor_stall_reward
        reward_shaping['quad_rewards']['room_floor'] = cfg.quads_room_floor_reward
        reward_shaping['quad_rewards']['room_wall'] = cfg.quads_room_wall_reward
        reward_shaping['quad_rewards']['room_ceiling'] = cfg.quads_room_ceiling_reward
        annealing = [
            AnnealSchedule(
                'quadcol_bin', cfg.quads_collision_reward, cfg.anneal_collision_steps,
                cfg.quads_collision_reward * initial_ratio,
            ),
            AnnealSchedule('quadcol_bin_smooth_max', cfg.quads_collision_smooth_max_penalty,
                           cfg.anneal_collision_steps,
                           cfg.quads_collision_smooth_max_penalty * initial_ratio),
            AnnealSchedule(
                'quadcol_bin_obst', cfg.quads_obst_collision_reward, cfg.anneal_collision_steps,
                cfg.quads_obst_collision_reward * initial_ratio,
            ),
            AnnealSchedule(
                'vel', cfg.quads_vel_reward, cfg.anneal_collision_steps,
                cfg.quads_vel_reward * initial_ratio,
            ),
        ]
    else:
        annealing = None

    env = QuadsRewardShapingWrapper(env, reward_shaping_scheme=reward_shaping, annealing=annealing,
                                    with_pbt=cfg.with_pbt)
    env = BoundedActionWrapper(env)
    env = QuadEnvCompatibility(env)
    env = ProjectionMapWrapper(
        env,
        enabled=cfg.visualize_projection_map or cfg.visualize_obstacle_point_cloud,
        show_obstacle_point_cloud=cfg.visualize_obstacle_point_cloud,
    )

    if cfg.visualize_v_value:
        raise ValueError("V-value map visualization is not included in this repository.")

    return env


def make_quadrotor_env(env_name, cfg=None, _env_config=None, render_mode=None, **kwargs):
    if env_name == 'quadrotor_multi':
        return make_quadrotor_env_multi(cfg, render_mode, **kwargs)
    else:
        raise NotImplementedError
