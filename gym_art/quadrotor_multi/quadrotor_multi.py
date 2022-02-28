import copy
from collections import deque

import numpy as np
import time
import gym

from copy import deepcopy

from gym_art.quadrotor_multi.params import obs_self_size_dict, obs_neighbor_size_dict
from gym_art.quadrotor_multi.quad_scenarios_utils import QUADS_MODE_GOAL_CENTERS, QUADS_MODE_OBST_INFO_LIST
from gym_art.quadrotor_multi.quad_utils import perform_collision_between_drones, perform_collision_with_obstacle, \
    calculate_collision_matrix, calculate_drone_proximity_penalties, calculate_obst_drone_proximity_penalties, \
    perform_collision_with_wall, perform_collision_with_ceiling, perform_collision_with_floor, perform_downwash

from gym_art.quadrotor_multi.quadrotor_multi_obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_single import GRAV, QuadrotorSingle
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quad_scenarios import create_scenario
from gym_art.quadrotor_multi.quad_obstacle_utils import OBSTACLES_SHAPE_LIST

EPS = 1E-6


class QuadrotorEnvMulti(gym.Env):
    def __init__(self,
                 num_agents, dynamics_params='DefaultQuad', dynamics_change=None, dynamics_randomize_every=None,
                 dyn_sampler_1=None, dyn_sampler_2=None, raw_control=True, raw_control_zero_middle=True, dim_mode='3D',
                 tf_control=False, sim_freq=200., sim_steps=2, obs_repr='xyz_vxyz_R_omega', ep_time=7, room_length=10,
                 room_width=10, room_height=10, init_random_state=False, rew_coeff=None, sense_noise=None,
                 verbose=False, gravity=GRAV, t2w_std=0.005, t2t_std=0.0005, excite=False,
                 dynamics_simplification=False, quads_mode='static_same_goal', quads_formation='circle_horizontal',
                 quads_formation_size=-1.0, swarm_obs='none', quads_use_numba=False, quads_settle=False,
                 quads_settle_range_meters=1.0, quads_vel_reward_out_range=0.8, quads_obstacle_mode='no_obstacles',
                 quads_view_mode='local', quads_obstacle_num=0, quads_obstacle_type='sphere', quads_obstacle_size=0.0,
                 collision_force=True, adaptive_env=False, obstacle_traj='gravity', local_obs=-1,
                 collision_hitbox_radius=2.0, collision_falloff_radius=2.0, collision_smooth_max_penalty=10.0,
                 local_metric='dist', local_coeff=0.0, use_replay_buffer=False, obstacle_obs_mode='relative',
                 obst_penalty_fall_off=10.0, vis_acc_arrows=False, viz_traces=25, viz_trace_nth_step=1,
                 local_obst_obs=-1, obst_enable_sim=True, obst_obs_type='none', quads_reward_ep_len=True, obst_level=-1,
                 obst_stack_num=4, enable_sim_room='none', obst_level_mode=0, obst_proximity_mode=0,
                 obst_inf_height=False, obst_level_change_cond=0.5, obst_collision_enable_grace_period=False,
                 crash_mode=0, clip_floor_vel_mode=0, midreset=False, crash_reset_threshold=200,
                 neighbor_rel_pos_mode=0, obst_rel_pos_mode=0, neighbor_prox_mode=0, obst_midreset=False,
                 obst_col_reset_threshold=1, print_info=False, apply_downwash=False, normalize_obs=False,
                 freeze_obst_level=False, obst_rel_pos_clip_value=2.0, one_pass_per_episode=False,
                 obst_level_crash_min=2.0, obst_level_crash_max=3.0, obst_level_col_obst_quad_min=2.0,
                 obst_level_col_obst_quad_max=4.0, obst_level_col_quad_min=0.5, obst_level_col_quad_max=1.0,
                 obst_level_pos_min=110.0, obst_level_pos_max=130.0, extra_crash_reward=False,
                 obst_generation_mode='random', use_pos_diff=False, obst_smooth_penalty_mode='linear', curriculum_min_obst=0,
                 nearest_nbrs=0):

        super().__init__()

        assert local_obs <= num_agents - 1 or local_obs == -1, \
            f'Invalid value ({local_obs}) passed to --local_obs. Should be 0 < n < num_agents - 1, or -1'
        if local_obs == -1:
            local_obs = num_agents - 1

        # Parameters for multi-drones
        self.num_agents = num_agents
        self.normalize_obs = normalize_obs
        # # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = (room_length, room_width, room_height)
        self.adaptive_env = adaptive_env
        self.quads_view_mode = quads_view_mode
        self.enable_sim_room = enable_sim_room
        # we don't actually create a scene object unless we want to render stuff
        self.scene = None

        # # Init multi drone envs
        self.envs = []
        for i in range(self.num_agents):
            e = QuadrotorSingle(
                dynamics_params=dynamics_params, dynamics_change=dynamics_change,
                dynamics_randomize_every=dynamics_randomize_every, dyn_sampler_1=dyn_sampler_1,
                dyn_sampler_2=dyn_sampler_2, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
                dim_mode=dim_mode, tf_control=tf_control, sim_freq=sim_freq, sim_steps=sim_steps, obs_repr=obs_repr,
                ep_time=ep_time, room_length=room_length, room_width=room_width, room_height=room_height,
                init_random_state=init_random_state, sense_noise=sense_noise, verbose=verbose, gravity=gravity,
                t2w_std=t2w_std, t2t_std=t2t_std, excite=excite, dynamics_simplification=dynamics_simplification,
                use_numba=quads_use_numba, swarm_obs=swarm_obs, num_agents=num_agents, quads_settle=quads_settle,
                quads_settle_range_meters=quads_settle_range_meters,
                quads_vel_reward_out_range=quads_vel_reward_out_range, view_mode=quads_view_mode,
                obstacle_mode=quads_obstacle_mode, obstacle_num=quads_obstacle_num, num_use_neighbor_obs=local_obs,
                num_local_obst=local_obst_obs, obst_obs_type=obst_obs_type, quads_reward_ep_len=quads_reward_ep_len,
                clip_floor_vel_mode=clip_floor_vel_mode, normalize_obs=normalize_obs, obst_inf_height=obst_inf_height,
                use_pos_diff=use_pos_diff, nearest_nbrs=nearest_nbrs
            )
            self.envs.append(e)

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space
        self.obs_mean = (self.observation_space.high + self.observation_space.low) / 2.0
        self.obs_std = (self.observation_space.high - self.observation_space.low) / 2.0
        # # Reward shaping
        self.extra_crash_reward = extra_crash_reward
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=0., quadcol_bin_smooth_max=0., quadcol_bin_obst=0., quadcol_bin_obst_smooth_max=0.0,
        )
        rew_coeff_orig = copy.deepcopy(self.rew_coeff)
        if rew_coeff is not None:
            assert isinstance(rew_coeff, dict)
            assert set(rew_coeff.keys()).issubset(set(self.rew_coeff.keys()))
            self.rew_coeff.update(rew_coeff)
        for key in self.rew_coeff.keys():
            self.rew_coeff[key] = float(self.rew_coeff[key])
        orig_keys = list(rew_coeff_orig.keys())
        # Checking to make sure we didn't provide some false rew_coeffs (for example by misspelling one of the params)
        assert np.all([key in orig_keys for key in self.rew_coeff.keys()])

        # # Aux variables for observation space of quads
        self.quad_arm = self.envs[0].dynamics.arm
        self.control_freq = self.envs[0].control_freq
        self.control_dt = 1.0 / self.control_freq
        self.pos = np.zeros([self.num_agents, 3])  # Matrix containing all positions
        self.quads_mode = quads_mode
        obs_self_size = obs_self_size_dict[obs_repr]

        # Aux variables for scenarios
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.set_room_dims,
                                        rew_coeff=self.rew_coeff, quads_formation=quads_formation,
                                        quads_formation_size=quads_formation_size,
                                        one_pass_per_episode=one_pass_per_episode)
        self.quads_formation_size = quads_formation_size
        self.goal_central = np.array([0., 0., 2.])

        # # Parameters for crashing with walls, floor or ceiling
        self.crash_mode = crash_mode

        # # Parameters for reset drones when they stay on the floor so long or collide with obstales many times
        # # count the continuous crash ticks, 100 ticks
        self.midreset = midreset
        self.all_crash_counter = np.zeros(self.num_agents)
        self.all_crash_threshold = crash_reset_threshold

        # # Parameters for print infomation
        self.print_info = print_info
        self.obst_counter_list = []
        self.cur_ep_obst_counter = 0
        self.real_obst_counter_list = []
        self.real_cur_ep_obst_counter = 0

        # Parameters for neighbor drones
        self.swarm_obs = swarm_obs
        self.num_use_neighbor_obs = local_obs
        self.local_metric = local_metric
        self.local_coeff = local_coeff
        self.neighbor_obs_size = obs_neighbor_size_dict[swarm_obs]
        self.nearest_nbrs = nearest_nbrs

        # # Clip neighbor observations
        clip_neighbor_space_length = obs_self_size + self.num_use_neighbor_obs * self.neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[obs_self_size:clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[obs_self_size:clip_neighbor_space_length]

        # # Parameters for Smooth collision function of neighbors
        self.neighbor_rel_pos_mode = neighbor_rel_pos_mode
        self.neighbor_prox_mode = neighbor_prox_mode

        # measuring the total number of pairwise collisions per episode
        self.collisions_per_episode = 0

        # some collisions may happen because the quadrotors get initialized on the collision course
        # if we wait a couple of seconds, then we can eliminate all the collisions that happen due to initialization
        # this is the actual metric that we want to minimize
        self.collisions_after_settle = 0
        self.collisions_grace_period_seconds = 1.5

        # collision proximity penalties
        self.collision_hitbox_radius = collision_hitbox_radius
        self.collision_falloff_radius = collision_falloff_radius
        self.collision_smooth_max_penalty = collision_smooth_max_penalty

        self.prev_drone_collisions, self.curr_drone_collisions = [], []
        self.all_collisions = {}
        self.apply_collision_force = collision_force

        # Parameters for Obstacles
        self.multi_obstacles = None
        self.obstacle_mode = quads_obstacle_mode
        self.obstacle_num = quads_obstacle_num
        self.obst_enable_sim = obst_enable_sim
        self.obst_proximity_mode = obst_proximity_mode
        self.obst_obs_type = obst_obs_type
        self.use_obstacles = self.obstacle_mode != 'no_obstacles' and self.obstacle_num > 0
        self.obst_inf_height = obst_inf_height
        self.obst_midreset = obst_midreset
        self.obst_col_reset_threshold = obst_col_reset_threshold
        self.obst_midreset_list = np.zeros(self.num_agents)
        self.obst_generation_mode = obst_generation_mode
        self.obst_change_step = 1.0
        self.obst_smooth_penalty_mode = obst_smooth_penalty_mode
        self.curriculum_min_obst = curriculum_min_obst

        # # Parameters used in controlling different level of obstacles (curriculum learning)
        self.freeze_obst_level = freeze_obst_level
        self.obst_level = obst_level
        self.obst_level_mode = obst_level_mode
        self.episode_num_control_level = 10
        self.obst_level_num_window = 2
        self.obst_split_level = 0
        self.obst_level_condition_dict = {
            'crash': {
                'low_bound': obst_level_crash_min,
                'high_bound': obst_level_crash_max,
                'cur_val': 0.0,
                'value_arr': deque([], maxlen=self.episode_num_control_level)
            },
            'collision_obst_quad': {
                'low_bound': obst_level_col_obst_quad_min,
                'high_bound': obst_level_col_obst_quad_max,
                'value_arr': deque([], maxlen=self.episode_num_control_level)
            },
        }
        self.episode_id = 0
        # # At the very start, since drones randomly initialized in the air, drones might collide with obstacles.
        # # Ignore penalties of smooth collision with obstacles and real collision penalties at the very beginning.
        self.obst_collision_enable_grace_period = obst_collision_enable_grace_period
        self.obst_collisions_grace_period_seconds = 1.0

        if self.use_obstacles:
            obstacle_max_init_vel = 4.0 * self.envs[0].max_init_vel
            obstacle_init_box = self.envs[0].box  # box of env is: 2 meters
            # This parameter is used to judge whether obstacles are out of room, and then, we can reset the obstacles
            self.obstacle_room = self.envs[0].room_box  # [[-5, -5, 0], [5, 5, 10]]
            dt = 1.0 / sim_freq
            if self.obstacle_mode.startswith('dynamic'):
                self.set_obstacles = np.zeros(self.obstacle_num, dtype=bool)
            else:
                self.set_obstacles = np.ones(self.obstacle_num, dtype=bool)
            self.obstacle_shape = quads_obstacle_type
            self.obst_penalty_fall_off = obst_penalty_fall_off
            self.multi_obstacles = MultiObstacles(
                mode=self.obstacle_mode, num_obstacles=self.obstacle_num, max_init_vel=obstacle_max_init_vel,
                init_box=obstacle_init_box, dt=dt, quad_size=self.quad_arm, shape=self.obstacle_shape,
                size=quads_obstacle_size, traj=obstacle_traj, obs_mode=obstacle_obs_mode, num_local_obst=local_obst_obs,
                obs_type=self.obst_obs_type, drone_env=self.envs[0], level=self.obst_level,
                stack_num=obst_stack_num, level_mode=obst_level_mode, inf_height=obst_inf_height,
                room_dims=self.room_dims, rel_pos_mode=obst_rel_pos_mode, rel_pos_clip_value=obst_rel_pos_clip_value,
                obst_level_num_window=self.obst_level_num_window, obst_generation_mode=obst_generation_mode,
                obst_change_step=self.obst_change_step
            )

            # collisions between obstacles and quadrotors
            self.obst_quad_collisions_per_episode = 0
            self.obst_quad_collisions_per_episode_after_settle = 0
            self.prev_obst_quad_collisions = []

        # Parameter for rendering
        self.simulation_start_time = 0
        self.frames_since_last_render = self.render_skip_frames = 0
        self.render_every_nth_frame = 1
        self.render_speed = 1.0  # set to below 1 slowmo, higher than 1 for fast forward (if simulator can keep up)

        # # Set to true whenever we need to reset the OpenGL scene in render()
        self.reset_scene = False
        self.vis_acc_arrows = vis_acc_arrows
        self.viz_traces = viz_traces
        self.viz_trace_nth_step = viz_trace_nth_step

        # Parameters for replay buffer
        self.use_replay_buffer = use_replay_buffer
        self.activate_replay_buffer = False  # only start using the buffer after the drones learn how to fly
        self.saved_in_replay_buffer = False  # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

        # Parameters for physical deployment
        self.apply_downwash = apply_downwash

    def set_room_dims(self, dims):
        # dims is a (x, y, z) tuple
        self.room_dims = dims

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def get_rel_pos_vel_item(self, env_id, indices=None):
        i = env_id

        if indices is None:
            # if not specified explicitly, consider all neighbors
            indices = [j for j in range(self.num_agents) if j != i]

        cur_pos = self.envs[i].dynamics.pos
        cur_vel = self.envs[i].dynamics.vel
        pos_neighbor = np.stack([self.envs[j].dynamics.pos for j in indices])
        vel_neighbor = np.stack([self.envs[j].dynamics.vel for j in indices])
        rot_neighbor = np.stack([self.envs[j].dynamics.rot.reshape(-1) for j in indices])
        omega_neighbor = np.stack([self.envs[j].dynamics.omega for j in indices])
        pos_rel = pos_neighbor - cur_pos
        if self.neighbor_rel_pos_mode == 1:
            pos_rel_mag = np.linalg.norm(pos_rel, axis=1)
            pos_rel_norm = np.array([pos_rel[i] / (pos_rel_mag[i] + 1e-6 if pos_rel_mag[i] == 0.0 else pos_rel_mag[i])
                                     for i in range(len(pos_rel_mag))])
            pos_rel = pos_rel - 2.0 * self.quad_arm * pos_rel_norm

        vel_rel = vel_neighbor - cur_vel
        return pos_rel, vel_rel, rot_neighbor, omega_neighbor

    def get_rel_pos_vel_stack(self):
        rel_pos_stack, rel_vel_stack = [], []
        rot_stack, omega_stack = [], []
        for i in range(self.num_agents):
            pos_rel, vel_rel, rot_neighbor, omega_neighbor = self.get_rel_pos_vel_item(env_id=i)
            rel_pos_stack.append(pos_rel)
            rel_vel_stack.append(vel_rel)
            rot_stack.append(rot_neighbor)
            omega_stack.append(omega_neighbor)

        return np.array(rel_pos_stack), np.array(rel_vel_stack), np.array(rot_stack), np.array(omega_stack)

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        i = env_id
        pos_neighbors_rel, vel_neighbors_rel, rot_neighbors, omega_neighbors = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])

        if self.swarm_obs == 'pos_vel':
            obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        elif self.swarm_obs == 'pos_vel_size':
            sizes = np.ones((pos_neighbors_rel.shape[0], 1)) * self.envs[0].dynamics.arm
            obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel, rot_neighbors, omega_neighbors, sizes), axis=1)
        else:
            neighbor_goals_rel = np.stack([self.envs[j].goal for j in closest_drones[i]]) - self.envs[i].dynamics.pos

            if self.swarm_obs == 'pos_vel_goals':
                obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel, neighbor_goals_rel), axis=1)
            elif self.swarm_obs == 'pos_vel_goals_ndist_gdist':
                dist_to_neighbors = np.linalg.norm(pos_neighbors_rel, axis=1).reshape(-1, 1)
                dist_to_neighbor_goals = np.linalg.norm(neighbor_goals_rel, axis=1).reshape(-1, 1)
                obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel, neighbor_goals_rel, dist_to_neighbors, dist_to_neighbor_goals), axis=1)
            else:
                raise NotImplementedError

        return obs_neighbor_rel

    def extend_obs_space(self, obs, closest_drones):
        assert self.swarm_obs in ['pos_vel', 'pos_vel_goals', 'pos_vel_size', 'pos_vel_goals_ndist_gdist'], \
            f'Invalid parameter {self.swarm_obs} passed in --obs_space'

        obs_neighbors = []
        for i in range(len(self.envs)):
            obs_neighbor_rel = self.get_obs_neighbor_rel(env_id=i, closest_drones=closest_drones)
            obs_neighbors.append(obs_neighbor_rel.reshape(-1))
        obs_neighbors = np.stack(obs_neighbors)

        # clip observation space of neighborhoods
        obs_neighbors = np.clip(
            obs_neighbors, a_min=self.clip_neighbor_space_min_box, a_max=self.clip_neighbor_space_max_box,
        )
        obs_ext = np.concatenate((obs, obs_neighbors), axis=1)
        return obs_ext

    def neighborhood_indices(self):
        """Return a list of closest drones for each drone in the swarm."""
        # indices of all the other drones except us
        indices = [[j for j in range(self.num_agents) if i != j] for i in range(self.num_agents)]
        indices = np.array(indices)

        if self.num_use_neighbor_obs == self.num_agents - 1:
            return indices
        elif 1 <= self.num_use_neighbor_obs < self.num_agents - 1:
            close_neighbor_indices = []

            for i in range(self.num_agents):
                rel_pos, rel_vel = self.get_rel_pos_vel_item(env_id=i, indices=indices[i])
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 0.01)
                rel_pos_unit = rel_pos / rel_dist[:, None]

                # new relative distance is a new metric that combines relative position and relative velocity
                # F = alpha * distance + (1 - alpha) * dot(normalized_direction_to_other_drone, relative_vel)
                if self.local_metric == "dist":
                    # the smaller the new_rel_dist, the closer the drones
                    new_rel_dist = rel_dist + self.local_coeff * np.sum(rel_pos_unit * rel_vel, axis=1)
                elif self.local_metric == "dist_inverse":
                    new_rel_dist = 1.0 / rel_dist - self.local_coeff * np.sum(rel_pos_unit * rel_vel, axis=1)
                    new_rel_dist = -1.0 * new_rel_dist
                else:
                    raise NotImplementedError(f"Unknown local metric {self.local_metric}")

                rel_pos_index = new_rel_dist.argsort()
                rel_pos_index = rel_pos_index[:self.num_use_neighbor_obs]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def add_neighborhood_obs(self, obs):
        if self.swarm_obs != 'none' and self.num_agents > 1:
            indices = self.neighborhood_indices()
            obs_ext = self.extend_obs_space(obs, closest_drones=indices)
            return obs_ext
        else:
            return obs

    def get_collision_penalty_by_vel(self, drone_pair):
        rew_quad_prox_vel_collision = np.zeros(self.num_agents)
        if len(drone_pair) == 0:
            return rew_quad_prox_vel_collision
        for pair in drone_pair:
            # get drone id
            d0 = pair[0]
            d1 = pair[1]

            # get vel
            v0 = self.envs[d0].dynamics.vel
            v1 = self.envs[d1].dynamics.vel

            # get speed
            s0 = np.linalg.norm(v0)
            s1 = np.linalg.norm(v1)

            # calculate penalty
            # p = 2.0 * (s0 * s1 - v0 * v1) = 2.0 * (s0 * s1 * (1 - cos(v0, v1)))
            penalty = -2.0 * (1.01 * s0 * s1 - np.dot(v0, v1))

            rew_quad_prox_vel_collision[d0] = penalty
            rew_quad_prox_vel_collision[d1] = penalty

        return rew_quad_prox_vel_collision

    def get_collision_obst_penalty_by_vel(self, drone_obst_pair):
        rew_drone_obst_prox_vel_collision = np.zeros(self.num_agents)
        if len(drone_obst_pair) == 0:
            return rew_drone_obst_prox_vel_collision

        for pair in drone_obst_pair:
            # get drone id
            drone_id = pair[0]
            obst_id = pair[1]

            # get drone vel & obst pos
            drone_vel = self.envs[drone_id].dynamics.vel

            drone_pos = self.envs[drone_id].dynamics.pos
            obst_pos = self.multi_obstacles.obstacles[obst_id].pos
            rel_pos = obst_pos - drone_pos

            rel_pos_size = np.linalg.norm(rel_pos)
            drone_speed = np.linalg.norm(self.envs[drone_id].dynamics.vel)
            cos_theta = np.dot(drone_vel, rel_pos) / (drone_speed * rel_pos_size)

            # calculate penalty
            # p = 2.0 * drone_speed * cos(rel_pos, d_vel)
            penalty = -2.0 * drone_speed * cos_theta

            rew_drone_obst_prox_vel_collision[drone_id] = penalty

        return rew_drone_obst_prox_vel_collision

    def neighbor_collision_info(self):
        # Calculating collisions between drones
        drone_col_matrix, self.curr_drone_collisions, distance_matrix = \
            calculate_collision_matrix(self.pos, self.quad_arm, self.collision_hitbox_radius)

        self.last_step_unique_collisions = np.setdiff1d(self.curr_drone_collisions, self.prev_drone_collisions)

        # collision between 2 drones counts as a single collision
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        if collisions_curr_tick > 0:
            if self.envs[0].tick >= self.collisions_grace_period_seconds * self.control_freq:
                self.collisions_after_settle += collisions_curr_tick

        self.prev_drone_collisions = self.curr_drone_collisions

        rew_collisions_raw = np.zeros(self.num_agents)
        rew_quad_prox_vel_collision = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
            # self.curr_drone_collisions
            if self.extra_crash_reward:
                rew_quad_prox_vel_collision = self.get_collision_penalty_by_vel(drone_pair=self.curr_drone_collisions)


        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        rew_proximity = -1.0 * calculate_drone_proximity_penalties(
            distance_matrix=distance_matrix, arm=self.quad_arm, dt=self.control_dt,
            penalty_fall_off=self.collision_falloff_radius,
            max_penalty=self.rew_coeff["quadcol_bin_smooth_max"],
            num_agents=self.num_agents, proximity_mode=self.neighbor_prox_mode
        )

        return drone_col_matrix, rew_collisions_raw, rew_collisions, rew_proximity, rew_quad_prox_vel_collision

    def drone_obst_collision_info(self):
        if self.use_obstacles:
            obst_quad_col_matrix, curr_obst_quad_collisions, curr_all_collisions, obst_quad_distance_matrix \
                = self.multi_obstacles.collision_detection(pos_quads=self.pos, set_obstacles=self.set_obstacles)
            obst_quad_last_step_unique_collisions = np.setdiff1d(curr_obst_quad_collisions,
                                                                 self.prev_obst_quad_collisions)

            self.obst_quad_collisions_per_episode += len(obst_quad_last_step_unique_collisions)
            self.prev_obst_quad_collisions = curr_obst_quad_collisions

            if len(obst_quad_last_step_unique_collisions) > 0:
                if self.envs[0].tick >= self.obst_collisions_grace_period_seconds * self.control_freq:
                    self.obst_quad_collisions_per_episode_after_settle += len(obst_quad_last_step_unique_collisions)

            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            rew_drone_obst_prox_vel_collision = np.zeros(self.num_agents)
            if len(obst_quad_last_step_unique_collisions) > 0:
                # We assign penalties to the drones which collide with the obstacles
                # And obst_quad_last_step_unique_collisions only include drones' id
                rew_obst_quad_collisions_raw[obst_quad_last_step_unique_collisions] = -1.0
                self.obst_midreset_list[obst_quad_last_step_unique_collisions] += 1.0
                if self.extra_crash_reward:
                    assert self.multi_obstacles.shape == 'cylinder'
                    rew_drone_obst_prox_vel_collision = self.get_collision_obst_penalty_by_vel(drone_obst_pair=
                                                                                               curr_all_collisions)

                for drone_id in obst_quad_last_step_unique_collisions:
                    if not self.envs[drone_id].crashed:
                        self.cur_ep_obst_counter += 1

                if self.print_info:
                    self.real_cur_ep_obst_counter += len(obst_quad_last_step_unique_collisions)

            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw

            # penalties for low distance between obstacles and drones
            rew_obst_quad_proximity = -1.0 * calculate_obst_drone_proximity_penalties(
                distance_matrix=obst_quad_distance_matrix, arm=self.quad_arm, dt=self.control_dt,
                penalty_fall_off=self.obst_penalty_fall_off,
                max_penalty=self.rew_coeff["quadcol_bin_obst_smooth_max"],
                num_agents=self.num_agents,
                proximity_mode=self.obst_proximity_mode,
                obst_smooth_penalty_mode=self.obst_smooth_penalty_mode
            )
        else:
            obst_quad_col_matrix = np.zeros((self.num_agents, self.obstacle_num))
            curr_all_collisions = []
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            rew_collisions_obst_quad = np.zeros(self.num_agents)
            rew_obst_quad_proximity = np.zeros(self.num_agents)
            rew_drone_obst_prox_vel_collision = np.zeros(self.num_agents)

        return obst_quad_col_matrix, curr_all_collisions, rew_obst_quad_collisions_raw, rew_collisions_obst_quad, \
               rew_obst_quad_proximity, rew_drone_obst_prox_vel_collision

    def simulate_colliding_with_room(self, drone_col_matrix, obst_quad_col_matrix):
        applied_room_collide_flag = False
        ground_collisions = np.array([env.dynamics.crashed_floor for env in self.envs])
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in self.envs])
        wall_collisions = np.array([env.dynamics.crashed_wall for env in self.envs])

        self.all_collisions = {'drone': np.sum(drone_col_matrix, axis=1), 'ground': ground_collisions,
                               'obstacle': np.sum(obst_quad_col_matrix, axis=1), 'ceiling': ceiling_collisions,
                               'wall': wall_collisions}

        if self.enable_sim_room != 'none':
            sim_list = self.enable_sim_room.split('-')
            if 'ceiling' in sim_list:
                ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
                if len(ceiling_crash_list) > 0:
                    applied_room_collide_flag = True
                for val in ceiling_crash_list:
                    perform_collision_with_ceiling(drone_dyn=self.envs[val].dynamics)
            if 'floor' in sim_list:
                floor_crash_list = np.where(ground_collisions >= 1)[0]
                if len(floor_crash_list) > 0:
                    applied_room_collide_flag = True
                for val in floor_crash_list:
                    perform_collision_with_floor(drone_dyn=self.envs[val].dynamics, crash_mode=self.crash_mode)
            if 'wall' in sim_list:
                wall_crash_list = np.where(wall_collisions >= 1)[0]
                if len(wall_crash_list) > 0:
                    applied_room_collide_flag = True
                for val in wall_crash_list:
                    perform_collision_with_wall(drone_dyn=self.envs[val].dynamics, room_box=self.envs[0].room_box,
                                                crash_mode=self.crash_mode)
        return applied_room_collide_flag

    def concatenate_obstacle_obs(self, obs):
        quads_vel = np.array([e.dynamics.vel for e in self.envs])

        if self.obstacle_mode == 'dynamic' and self.obstacle_num > 0:
            tmp_obs = self.multi_obstacles.step(obs=obs, quads_pos=self.pos, quads_vel=quads_vel,
                                                set_obstacles=self.set_obstacles)

            # If there are still at least one obstacle flying in the air, we should check the function below
            # and reset the counter only if all obstacles hit the floor
            if self.obstacle_mode.startswith('dynamic') and self.set_obstacles.any():
                for obst_i, obstacle in enumerate(self.multi_obstacles.obstacles):
                    if not self.set_obstacles[obst_i]:
                        continue

                    if self.multi_obstacles.obstacles[obst_i].tmp_traj == "gravity":
                        obstacle_pos = copy.deepcopy(obstacle.pos)
                        obstacle_pos[2] = obstacle.pos[2] - obstacle.size / 2
                        if not np.array_equal(obstacle_pos,
                                              np.clip(obstacle_pos,
                                                      a_min=self.obstacle_room[0],  # [-5, -5, 0]
                                                      a_max=self.obstacle_room[1])):  # [5, 5, 10]
                            self.set_obstacles[obst_i] = False
                            self.quads_formation_size = self.scenario.formation_size
                            self.goal_central = np.mean(self.scenario.goals, axis=0)
                            obst_shape = obstacle.shape
                            if self.obstacle_shape == 'random':
                                obst_shape_id = np.random.randint(low=0, high=len(OBSTACLES_SHAPE_LIST))
                                obst_shape = OBSTACLES_SHAPE_LIST[obst_shape_id]

                            obstacle.reset(set_obstacle=False, formation_size=self.quads_formation_size,
                                           goal_central=self.goal_central, shape=obst_shape, quads_pos=self.pos,
                                           quads_vel=quads_vel)
                    elif self.multi_obstacles.obstacles[obst_i].tmp_traj == "electron":
                        tick = self.envs[0].tick
                        control_step_for_sec = int(7.0 * self.control_freq)
                        if tick % control_step_for_sec == 0 and tick > 0:
                            self.set_obstacles[obst_i] = False

            if not self.set_obstacles.all():
                self.set_obstacles = np.ones(self.obstacle_num, dtype=bool)
                self.quads_formation_size = self.scenario.formation_size
                self.goal_central = np.mean(self.scenario.goals, axis=0)
                tmp_obs = self.multi_obstacles.reset(
                    obs=obs, quads_pos=self.pos, quads_vel=quads_vel, set_obstacles=self.set_obstacles,
                    formation_size=self.quads_formation_size, goal_central=self.goal_central)

                # In testing mode, which means reset the scene, and this in visualization would make people feel
                # the screen stuck around one second
                if self.obstacle_num > 1:
                    self.reset_scene = True

            obs = tmp_obs
        elif self.obstacle_mode.startswith('static'):
            obs = self.multi_obstacles.step(obs=obs, quads_pos=self.pos, quads_vel=quads_vel,
                                            set_obstacles=self.set_obstacles)

        return obs

    def can_drones_fly(self):
        """
        Here we count the average number of collisions with the walls and ground in the last N episodes
        Returns: True if drones are considered proficient at flying
        """
        res = abs(np.mean(self.crashes_in_recent_episodes)) < 1 and len(self.crashes_in_recent_episodes) >= 10
        return res

    def init_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        self.scene = Quadrotor3DSceneMulti(
            models=models,
            w=640, h=480, resizable=True, multi_obstacles=self.multi_obstacles, viewpoint=self.envs[0].viewpoint,
            obstacle_mode=self.obstacle_mode, room_dims=self.room_dims, num_agents=self.num_agents,
            render_speed=self.render_speed, formation_size=self.quads_formation_size,
            vis_acc_arrows=self.vis_acc_arrows, viz_traces=self.viz_traces, viz_trace_nth_step=self.viz_trace_nth_step,
            obst_inf_height=self.obst_inf_height
        )

    def add_extra_info(self, info):
        info['num_obst_in_room'] = self.multi_obstacles.obst_num_in_room
        info['max_obst_num'] = self.multi_obstacles.max_obst_num
        info['obst_change_step'] = self.obst_change_step
        return info

    def change_level(self):
        if self.freeze_obst_level:
            return
        if self.obst_level_mode == 0 and self.obst_level > -1:
            return

        collected_episode_num = max(len(self.obst_level_condition_dict['crash']['value_arr']),
                                    len(self.obst_level_condition_dict['collision_obst_quad']['value_arr']))

        if collected_episode_num < self.episode_num_control_level:
            return

        mean_crash = abs(np.mean(self.obst_level_condition_dict['crash']['value_arr']))
        mean_collision_obst_quad = np.mean(self.obst_level_condition_dict['collision_obst_quad']['value_arr'])

        # upgrade level
        metric_obst_quad = 1 + np.clip(self.obst_level, a_min=0, a_max=8) * 0.12  # The metric of moving to the next level is in [1, 3]
        if self.obst_level < 0:  # -1
            upgrade_flag = [
                mean_crash < 3.0,
            ]
        else:
            upgrade_flag = [
                mean_collision_obst_quad < metric_obst_quad,
            ]

        if all(upgrade_flag):
            self.obst_level += 1
            self.obst_level = np.clip(self.obst_level, a_min=-1, a_max=self.obstacle_num - self.curriculum_min_obst)
            self.obst_level_condition_dict['crash']['value_arr'] = deque([], maxlen=self.episode_num_control_level)
            self.obst_level_condition_dict['collision_obst_quad']['value_arr'] = deque([], maxlen=self.episode_num_control_level)
            return

        # downgrade level
        downgrade_flag = [
            mean_collision_obst_quad > 2.0 * metric_obst_quad,
        ]

        if self.obst_level > -1 and any(downgrade_flag):
            self.obst_level -= 1
            self.obst_level = np.clip(self.obst_level, a_min=-1, a_max=self.obstacle_num - self.curriculum_min_obst)
            self.obst_level_condition_dict['crash']['value_arr'] = deque([], maxlen=self.episode_num_control_level)
            self.obst_level_condition_dict['collision_obst_quad']['value_arr'] = deque([], maxlen=self.episode_num_control_level)
            return

        return

    def reset_crashed_drones(self):
        if self.midreset:
            all_kinds_collisions = np.array([env.crashed for env in self.envs])
            if any(all_kinds_collisions):
                for i, e in enumerate(self.envs):
                    if all_kinds_collisions[i]:
                        self.all_crash_counter[i] += 1
                    else:
                        self.all_crash_counter[i] = 0

                    if self.all_crash_counter[i] > self.all_crash_threshold:
                        e.reset(midreset=True)
                        self.all_crash_counter[i] = 0
            else:
                self.all_crash_counter = np.zeros(self.num_agents)

        if self.obst_midreset:
            if any(self.obst_midreset_list):
                for i, e in enumerate(self.envs):
                    if self.obst_midreset_list[i] >= self.obst_col_reset_threshold:
                        e.reset(midreset=True)
                        self.obst_midreset_list[i] = 0

    def sort_obs(self, obs, self_obs_dim=24, nbr_obs_dim=19):
        '''
        Sort for the K nearest neighbor 'obstacles', where an object can be a drone or a pillar
        Discard the other neighbor observations
        '''
        nbr_obs = obs[:, self_obs_dim:]
        nbr_obs = nbr_obs.reshape(nbr_obs.shape[0], -1, nbr_obs_dim)
        all_neighbor_obs_size = nbr_obs_dim * self.nearest_nbrs
        for i in range(nbr_obs.shape[0]):
            # sort all neighbor objects according to their distances
            nbr_obs_i = nbr_obs[i]
            dists = np.linalg.norm(nbr_obs_i[:, :3], axis=1)
            inds_sorted = np.argsort(dists)
            nbr_obs[i] = nbr_obs[i, inds_sorted, :]
        nbr_obs = nbr_obs.reshape(nbr_obs.shape[0], -1)
        nbr_obs = nbr_obs[:, :all_neighbor_obs_size]
        obs = obs[:, :self_obs_dim + all_neighbor_obs_size]
        obs[:, self_obs_dim:] = nbr_obs
        return obs

    def reset_given_obst_scenario(self):
        if self.scenario.quads_mode == 'mix':
            scenario_mode = self.scenario.scenario_mode
        else:
            scenario_mode = self.scenario.quads_mode

        if scenario_mode not in QUADS_MODE_GOAL_CENTERS:
            spawn_flag = self.scenario.spawn_flag
            goal_start_point = self.scenario.start_point
            goal_end_point = self.scenario.end_point
            goal_points = []
        else:
            spawn_flag = -1
            goal_start_point = self.scenario.goals_center_list
            goal_end_point = copy.deepcopy(goal_start_point)
            if self.scenario.quads_mode == 'mix':
                goal_points = self.scenario.multi_goals
            else:
                goal_points = self.scenario.goals

        obst_num_in_room = self.scenario.obst_num_in_room

        return scenario_mode, spawn_flag, goal_start_point, goal_end_point, goal_points, obst_num_in_room

    def reset(self):
        obs, rewards, dones, infos = [], [], [], []
        if self.scenario.quads_mode in QUADS_MODE_OBST_INFO_LIST and self.use_obstacles:
            self.scenario.reset(obst_level=self.obst_level, obst_level_num_window=self.obst_level_num_window)
        elif self.scenario.quads_mode == 'mix' and self.use_obstacles:
            self.scenario.reset(obst_level=self.obst_level, obst_level_num_window=self.obst_level_num_window,
                                obst_num=self.obstacle_num, max_obst_num=self.multi_obstacles.max_obst_num,
                                obst_level_mode=self.obst_level_mode, curriculum_min_obst=self.curriculum_min_obst)
        else:
            self.scenario.reset()

        self.quads_formation_size = self.scenario.formation_size
        self.goal_central = np.mean(self.scenario.goals, axis=0)
        if self.use_obstacles:
            scenario_mode, spawn_flag, goal_start_point, goal_end_point, goal_points, obst_num_in_room\
                = self.reset_given_obst_scenario()

        # try to activate replay buffer if enabled
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_in_recent_episodes.append(self.crashes_last_episode)
            self.activate_replay_buffer = self.can_drones_fly()

        if self.adaptive_env:
            # TODO: introduce logic to choose the new room dims i.e. based on statistics from last N episodes, etc
            # e.g. self.room_dims = ....
            new_length, new_width, new_height = np.random.randint(1, 31, 3)
            self.room_dims = (new_length, new_width, new_height)

        for i, e in enumerate(self.envs):
            e.goal = self.scenario.goals[i]
            e.rew_coeff = self.rew_coeff
            e.update_env(*self.room_dims)
            if self.use_obstacles:
                if scenario_mode in QUADS_MODE_GOAL_CENTERS:
                    e.reset_spawn_flag_and_start_point(spawn_flag=spawn_flag, goal_start_point=goal_points[i])
                else:
                    e.reset_spawn_flag_and_start_point(spawn_flag=spawn_flag, goal_start_point=goal_start_point)

            observation = e.reset()
            obs.append(observation)

        # extend obs to see neighbors
        obs = self.add_neighborhood_obs(obs)

        # Reset Obstacles
        if self.use_obstacles:
            if self.obstacle_mode.startswith('dynamic'):
                self.set_obstacles = np.zeros(self.obstacle_num, dtype=bool)
            else:
                self.set_obstacles = np.ones(self.obstacle_num, dtype=bool)

            quads_pos = np.array([e.dynamics.pos for e in self.envs])
            quads_vel = np.array([e.dynamics.vel for e in self.envs])
            obs = self.multi_obstacles.reset(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel,
                                             set_obstacles=self.set_obstacles, formation_size=self.quads_formation_size,
                                             goal_central=self.goal_central, level=self.obst_level,
                                             goal_start_point=goal_start_point, goal_end_point=goal_end_point,
                                             scenario_mode=scenario_mode, obst_num_in_room=obst_num_in_room)
            self.obst_quad_collisions_per_episode = 0
            self.obst_quad_collisions_per_episode_after_settle = 0
            self.prev_obst_quad_collisions = []
            self.obst_midreset_list = np.zeros(self.num_agents)
            self.cur_ep_obst_counter = 0

        self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in
                               ['drone', 'ground', 'obstacle', 'wall', 'ceiling']}

        self.collisions_per_episode = self.collisions_after_settle = 0
        self.prev_drone_collisions, self.curr_drone_collisions = [], []

        self.reset_scene = True
        self.crashes_last_episode = 0

        obs = self.sort_obs(obs)
        if self.normalize_obs:
            # obs = np.clip(obs, a_min=self.observation_space.low, a_max=self.observation_space.high)
            obs = (obs - self.obs_mean) / self.obs_std

        return obs

    # noinspection PyTypeChecker
    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            observation, reward, done, info = self.envs[i].step(a)
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.pos[i, :] = self.envs[i].dynamics.pos

        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        if self.obst_generation_mode == 'cube':
            infos[0] = self.add_extra_info(info=infos[0])
        # Deal with collisions b/w drones
        # # 1. collision matrix of drones, 2. raw collision penalties, 3.  collision penalties
        # # 4. smooth penalties when drones are close to each other
        applied_force_flag = False
        drone_col_matrix, rew_collisions_raw, rew_collisions, rew_proximity, rew_quad_prox_vel_collision\
            = self.neighbor_collision_info()

        # COLLISION BETWEEN QUAD AND OBSTACLE(S)
        obst_quad_col_matrix, curr_all_collisions, rew_obst_quad_collisions_raw, rew_collisions_obst_quad, \
        rew_obst_quad_proximity, rew_drone_obst_prox_vel_collision = self.drone_obst_collision_info()

        # Apply force to change vel and omega: 1. Downwash; 2. Room; 3. b/w drones 4. b/w obstacles & drones
        # # 1. Apply random force of downwash
        if self.apply_downwash:
            envs_dynamics = [env.dynamics for env in self.envs]
            applied_downwash_flag = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            applied_force_flag = applied_force_flag or applied_downwash_flag

        # # 2. Simulate collisions when drones collide with the room
        applied_room_collide_flag = self.simulate_colliding_with_room(drone_col_matrix=drone_col_matrix,
                                                                      obst_quad_col_matrix=obst_quad_col_matrix)
        applied_force_flag = applied_force_flag or applied_room_collide_flag

        # # 3 & 4. Simulate collisions when drones collide with each other or with obstacles
        if self.apply_collision_force:
            if len(self.curr_drone_collisions) > 0 or len(curr_all_collisions) > 0:
                applied_drone_or_obst_flag = True
                applied_force_flag = applied_force_flag or applied_drone_or_obst_flag
            for val in self.curr_drone_collisions:
                perform_collision_between_drones(self.envs[val[0]].dynamics, self.envs[val[1]].dynamics)
            for val in curr_all_collisions:
                if self.obst_enable_sim:
                    perform_collision_with_obstacle(
                        drone_dyn=self.envs[val[0]].dynamics, obstacle_dyn=self.multi_obstacles.obstacles[val[1]],
                        quad_arm=self.quad_arm, room_dims=self.room_dims, inf_height=self.obst_inf_height,
                        crash_mode=self.crash_mode)

        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]
            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]

            rewards[i] += rew_proximity[i]
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]

            rewards[i] += rew_quad_prox_vel_collision[i]
            infos[i]["rewards"]["rew_quad_proximity_by_vel"] = rew_quad_prox_vel_collision[i]

            if self.use_obstacles:
                rewards[i] += rew_collisions_obst_quad[i]
                rewards[i] += rew_obst_quad_proximity[i]
                rewards[i] += rew_drone_obst_prox_vel_collision[i]

                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]
                infos[i]["rewards"]["rew_obst_quad_proximity"] = rew_obst_quad_proximity[i]

                infos[i]["rewards"]["rew_obst_quad_proximity_by_vel"] = rew_drone_obst_prox_vel_collision[i]

        # Run the scenario passed to self.quads_mode
        if self.scenario.quads_mode in QUADS_MODE_OBST_INFO_LIST or (
                self.scenario.quads_mode == 'mix' and self.scenario.scenario.quads_mode in QUADS_MODE_OBST_INFO_LIST):
            infos[0]['obstacles'] = self.multi_obstacles.obstacles
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=self.pos)

        # Adjust obs, specifically observation of drone itself after appplying random force
        # # 1. Downwash; 2. Room; 3. b/w drones 4. b/w obstacles & drones only changes vel and omega
        if applied_force_flag:
            obs = [e.state_vector(e) for e in self.envs]

        # Concatenate observations of neighbor drones and obstacles
        obs = self.add_neighborhood_obs(obs=obs)
        if self.use_obstacles:
            obs = self.concatenate_obstacle_obs(obs=obs)

        obs = self.sort_obs(obs)
        # Reset a drone if it's crash time on the floor larger than a predefined threshold
        self.reset_crashed_drones()

        # Used to check if we need to increase the level or decrease the level
        self.obst_level_condition_dict['crash']['cur_val'] += np.mean([infos[i]["rewards"]["rewraw_crash"] for i in range(self.num_agents)])

        # DONES
        if any(dones):
            if self.print_info:
                self.obst_counter_list.append(self.cur_ep_obst_counter)
                self.real_obst_counter_list.append(self.real_cur_ep_obst_counter)
                print("ep_num: ", len(self.obst_counter_list))
                print("counter_list: ", self.obst_counter_list)
                print("mean: ", np.mean(self.obst_counter_list))
                print("real counter_list: ", self.real_obst_counter_list)
                print("mean: ", np.mean(self.real_obst_counter_list))
                self.real_cur_ep_obst_counter = 0

            self.episode_id += 1
            self.obst_level_condition_dict['crash']['value_arr'].append(self.obst_level_condition_dict['crash']['cur_val'])
            self.obst_level_condition_dict['collision_obst_quad']['value_arr'].append(self.cur_ep_obst_counter)

            self.change_level()

            for i in range(len(infos)):
                if self.saved_in_replay_buffer:
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions_replay': self.collisions_per_episode,
                    }
                else:
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions': self.collisions_per_episode,
                        'num_collisions_after_settle': self.collisions_after_settle,
                        f'num_collisions_{self.scenario.name()}': self.collisions_after_settle,
                    }
                    if self.use_obstacles:
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = \
                            self.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats'][f'num_collisions_obst_{self.scenario.name()}'] = \
                            self.obst_quad_collisions_per_episode

                        infos[i]['episode_extra_stats']['after_settle_num_collisions_obst_quad'] = \
                            self.obst_quad_collisions_per_episode_after_settle
                        infos[i]['episode_extra_stats'][f'after_settle_num_collisions_obst_{self.scenario.name()}'] = \
                            self.obst_quad_collisions_per_episode_after_settle

                        infos[i]['episode_extra_stats']['episode_id'] = self.episode_id
                        infos[i]['episode_extra_stats']['obst_level'] = self.obst_level

                        infos[i]['episode_extra_stats']['overall_obst_counter_air'] = self.cur_ep_obst_counter
                        infos[i]['episode_extra_stats'][f'obst_counter_air_{self.scenario.name()}'] = self.cur_ep_obst_counter

            obs = self.reset()
            dones = [True] * len(dones)  # terminate the episode for all "sub-envs"

        # Normalize observations
        if self.normalize_obs:
            # obs = np.clip(obs, a_min=self.observation_space.low, a_max=self.observation_space.high)
            obs = (obs - self.obs_mean) / self.obs_std

        return obs, rewards, dones, infos

    def render(self, mode='human', verbose=False):
        models = tuple(e.dynamics.model for e in self.envs)

        if self.scene is None:
            self.init_scene_multi()

        if self.reset_scene:
            self.scene.update_models(models)
            self.scene.formation_size = self.quads_formation_size
            self.scene.update_env(self.room_dims)

            self.scene.reset(tuple(e.goal for e in self.envs), self.all_dynamics(), self.multi_obstacles, self.all_collisions)

            self.reset_scene = False

        if self.quads_mode == "mix":
            self.scene.formation_size = self.scenario.scenario.formation_size
        else:
            self.scene.formation_size = self.scenario.formation_size
        self.frames_since_last_render += 1

        if self.render_skip_frames > 0:
            self.render_skip_frames -= 1
            return None

        # this is to handle the 1st step of the simulation that will typically be very slow
        if self.simulation_start_time > 0:
            simulation_time = time.time() - self.simulation_start_time
        else:
            simulation_time = 0

        realtime_control_period = 1 / self.control_freq

        render_start = time.time()
        goals = tuple(e.goal for e in self.envs)
        self.scene.render_chase(all_dynamics=self.all_dynamics(), goals=goals, collisions=self.all_collisions,
                                mode=mode, multi_obstacles=self.multi_obstacles)
        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            self.scenario.scenario.update_formation_size(self.scene.formation_size)
        else:
            self.scenario.update_formation_size(self.scene.formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if mode == "human" and time_to_sleep > 0:
            time.sleep(time_to_sleep)

        if simulation_time + render_time > desired_time_between_frames:
            self.render_every_nth_frame += 1
            if verbose:
                print(f"Last render + simulation time {render_time + simulation_time:.3f}")
                print(f"Rendering does not keep up, rendering every {self.render_every_nth_frame} frames")
        elif simulation_time + render_time < realtime_control_period * (
                self.frames_since_last_render - 1) / self.render_speed:
            self.render_every_nth_frame -= 1
            if verbose:
                print(f"We can increase rendering framerate, rendering every {self.render_every_nth_frame} frames")

        if self.render_every_nth_frame > 5:
            self.render_every_nth_frame = 5
            if self.envs[0].tick % 20 == 0:
                print(f"Rendering cannot keep up! Rendering every {self.render_every_nth_frame} frames")

        self.render_skip_frames = self.render_every_nth_frame - 1
        self.frames_since_last_render = 0

        self.simulation_start_time = time.time()

    def __deepcopy__(self, memo):
        """OpenGL scene can't be copied naively."""

        cls = self.__class__
        copied_env = cls.__new__(cls)
        memo[id(self)] = copied_env

        # this will actually break the reward shaping functionality in PBT, but we need to fix it in SampleFactory, not here
        skip_copying = {"scene", "reward_shaping_interface"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_env, k, deepcopy(v, memo))

        # warning! deep-copied env has its scene uninitialized! We gotta reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
