import copy
import time
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np

from gym_art.quadrotor_multi.aerodynamics.downwash import perform_downwash
from gym_art.quadrotor_multi.collisions.obstacles import perform_collision_with_obstacle
from gym_art.quadrotor_multi.collisions.quadrotors import calculate_collision_matrix, \
    calculate_drone_proximity_penalties, perform_collision_between_drones, calculate_drone_obst_proximity_penalties
from gym_art.quadrotor_multi.collisions.room import perform_collision_with_wall, perform_collision_with_ceiling
from gym_art.quadrotor_multi.quad_utils import QUADS_OBS_REPR, QUADS_NEIGHBOR_OBS_TYPE

from gym_art.quadrotor_multi.obstacles.obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quadrotor_single import QuadrotorSingle
from gym_art.quadrotor_multi.scenarios.mix import create_scenario


class QuadrotorEnvMulti(gym.Env):
    def __init__(self, num_agents, ep_time, rew_coeff, obs_repr, obs_rel_rot, dynamic_goal,
                 # Neighbor
                 neighbor_visible_num, neighbor_obs_type, collision_hitbox_radius, collision_falloff_radius,
                 neighbor_obs_update_freq,
                 # Obstacle
                 use_obstacles, obst_density, obst_size, obst_spawn_area, obst_obs_type, obst_noise, grid_size,
                 obst_tof_resolution, obst_spawn_center, obst_grid_size_random, obst_grid_size_range, critic_rnn_size,
                 obst_critic_obs, obst_min_gap_threshold,
                 # Aerodynamics, Numba Speed Up, Scenarios, Room, Replay Buffer, Rendering
                 use_downwash, z_overlap, use_numba, quads_mode, sim2real_scenario, room_dims, use_replay_buffer, quads_view_mode,
                 quads_render,
                 # Quadrotor Specific (Do Not Change)
                 dynamics_params, raw_control, raw_control_zero_middle,
                 dynamics_randomize_every, dynamics_change, dyn_sampler_1,
                 sense_noise, init_random_state, use_ctbr,
                 # Rendering
                 render_mode='human',
                 # SBC
                 enable_sbc=False, sbc_neighbor_range=2.0, sbc_obst_range=2.0, sbc_obst_agg=0.2,
                 # Randomization
                 obst_density_random=False, obst_density_min=0.2, obst_density_max=0.8, obst_size_random=False,
                 obst_size_min=0.2, obst_size_max=0.4
                 ):
        super().__init__()

        # Predefined Parameters
        self.num_agents = num_agents
        obs_self_size = QUADS_OBS_REPR[obs_repr]
        if neighbor_visible_num == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = neighbor_visible_num

        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = room_dims
        self.quads_view_mode = quads_view_mode
        self.sim2real_scenario = sim2real_scenario

        # Generate All Quadrotors
        self.envs = []
        if obst_grid_size_random:
            min_grid_size = obst_grid_size_range[0]
        else:
            min_grid_size = grid_size

        if obst_density_random:
            max_obst_num = int(obst_density_max * (obst_spawn_area[0] / min_grid_size) * (obst_spawn_area[1] / min_grid_size))
        else:
            max_obst_num = int(obst_density * (obst_spawn_area[0] / min_grid_size) * (obst_spawn_area[1] / min_grid_size))

        for i in range(self.num_agents):
            e = QuadrotorSingle(
                # Quad Parameters
                dynamics_params=dynamics_params, dynamics_change=dynamics_change,
                dynamics_randomize_every=dynamics_randomize_every, dyn_sampler_1=dyn_sampler_1,
                raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle, sense_noise=sense_noise,
                init_random_state=init_random_state, obs_repr=obs_repr, ep_time=ep_time, room_dims=room_dims,
                use_numba=use_numba, obs_rel_rot=obs_rel_rot, dynamic_goal=dynamic_goal,
                # Neighbor
                num_agents=num_agents,
                neighbor_obs_type=neighbor_obs_type, num_use_neighbor_obs=self.num_use_neighbor_obs,
                # Obstacle
                use_obstacles=use_obstacles, obst_obs_type=obst_obs_type, obst_tof_resolution=obst_tof_resolution,
                obst_spawn_area=obst_spawn_area, obst_num=max_obst_num, critic_rnn_size=critic_rnn_size,
                obst_critic_obs=obst_critic_obs,
                #Controller
                use_ctbr=use_ctbr, use_sbc=enable_sbc, sbc_obst_agg=sbc_obst_agg
            )
            self.envs.append(e)

        # Set Obs & Act
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Aux variables
        self.quad_arm = self.envs[0].dynamics.arm
        self.control_freq = self.envs[0].control_freq
        self.control_dt = 1.0 / self.control_freq
        self.pos = np.zeros([self.num_agents, 3])
        self.vel = np.zeros([self.num_agents, 3])
        self.omega = np.zeros([self.num_agents, 3])
        self.rel_pos = np.zeros((self.num_agents, self.num_agents, 3))
        self.rel_vel = np.zeros((self.num_agents, self.num_agents, 3))
        self.low_h_count = 0

        # Reward
        self.rew_coeff = dict(
            # self
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., omega=1., rot=0., attitude=0.,
            spin=0.1, vel=0.,
            # collision
            quadcol_bin=5., quadcol_bin_smooth_max=4., quadcol_bin_obst=5., quads_obst_collision_prox_weight=0.0, 
            quads_obst_collision_prox_max=0.5, quads_obst_collision_prox_min=0.,
            # SBC
            sbc_acc=0.0, sbc_boundary=0.0
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

        # Neighbors
        neighbor_obs_size = QUADS_NEIGHBOR_OBS_TYPE[neighbor_obs_type]
        self.neighbor_obs_type = neighbor_obs_type

        self.clip_neighbor_space_length = self.num_use_neighbor_obs * neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[
                                           obs_self_size:obs_self_size + self.clip_neighbor_space_length]

        self.neighbor_pos_copy = copy.deepcopy(self.pos)
        self.neighbor_vel_copy = copy.deepcopy(self.vel)
        self.neighbor_obs_update_freq = neighbor_obs_update_freq
        # Obstacles
        self.use_obstacles = use_obstacles
        self.obst_obs_type = obst_obs_type
        self.obst_noise = obst_noise
        self.obstacles = None
        if self.use_obstacles:
            self.prev_obst_quad_collisions = []
            self.obst_quad_collisions_per_episode = 0
            self.obst_quad_collisions_after_settle = 0
            self.curr_quad_col = []
            self.obst_density = obst_density
            self.obst_spawn_area = obst_spawn_area
            self.obst_size = obst_size
            self.grid_size = grid_size
            assert self.obst_size <= self.grid_size
            self.obst_spawn_center = obst_spawn_center
            self.obst_grid_size_random = obst_grid_size_random
            self.obst_grid_size_range = obst_grid_size_range
            self.critic_rnn_size = critic_rnn_size
            self.obst_critic_obs = obst_critic_obs
            self.obst_tof_resolution = obst_tof_resolution
            # Randomization
            self.obst_density_random = obst_density_random
            self.obst_density_min = obst_density_min
            self.obst_density_max = obst_density_max

            self.obst_size_random = obst_size_random
            self.obst_size_min = obst_size_min
            self.obst_size_max = obst_size_max

            self.min_gap_threshold = obst_min_gap_threshold

            obst_params = {
                # Drone
                'quad_radius': self.quad_arm,
                # Obst
                'obst_obs_type': self.obst_obs_type,
                'obst_noise': self.obst_noise,
                'obst_tof_resolution': self.obst_tof_resolution,
                'critic_rnn_size': self.critic_rnn_size,
                'obst_critic_obs': self.obst_critic_obs,
                # Randomization
                'obst_grid_size_random': self.obst_grid_size_random,
                'obst_grid_size_range': self.obst_grid_size_range,
                'obst_grid_size': self.grid_size,

                'obst_density_random': self.obst_density_random,
                'obst_density_min': self.obst_density_min,
                'obst_density_max': self.obst_density_max,
                'obst_density': self.obst_density,

                'obst_size_random': self.obst_size_random,
                'obst_min_gap_threshold': self.min_gap_threshold,
                'obst_size_min': self.obst_size_min,
                'obst_size_max': self.obst_size_max,
                'obst_size': self.obst_size,

                'obst_spawn_area': self.obst_spawn_area,
                'obst_spawn_center': self.obst_spawn_center,
                'room_dims': self.room_dims,
                'sim2real_scenario': self.sim2real_scenario,
            }
            self.obstacles = MultiObstacles(params=obst_params)

        # Scenarios
        self.quads_mode = quads_mode
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=num_agents,
                                        room_dims=room_dims)

        # Collisions
        # # Collisions: Neighbors
        self.collisions_per_episode = 0
        # # # Ignore collisions because of spawn
        self.collisions_after_settle = 0
        self.collisions_grace_period_steps = 1.5 * self.control_freq
        self.collisions_grace_period_seconds = 1.5
        self.prev_drone_collisions = []

        self.collisions_final_grace_period_steps = 5.0 * self.control_freq
        self.collisions_final_5s = 0

        # # # Dense reward info
        self.collision_threshold = collision_hitbox_radius * self.quad_arm
        self.collision_falloff_threshold = collision_falloff_radius * self.quad_arm

        # # Collisions: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = 0
        self.collisions_wall_per_episode = 0
        self.collisions_ceiling_per_episode = 0

        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # Replay
        self.use_replay_buffer = use_replay_buffer
        # # only start using the buffer after the drones learn how to fly
        self.activate_replay_buffer = False
        # # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.saved_in_replay_buffer = False
        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

        # Numba
        self.use_numba = use_numba

        # Aerodynamics
        self.use_downwash = use_downwash
        self.z_overlap = z_overlap

        # Rendering
        # # set to true whenever we need to reset the OpenGL scene in render()
        self.render_mode = render_mode
        self.quads_render = quads_render
        self.scenes = []
        if self.quads_render:
            self.reset_scene = False
            self.simulation_start_time = 0
            self.frames_since_last_render = self.render_skip_frames = 0
            self.render_every_nth_frame = 1
            # # Use this to control rendering speed
            self.render_speed = 1.0
            self.quads_formation_size = 2.0
            self.all_collisions = {}

        # Log
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.distance_to_goal_xy = [[] for _ in range(len(self.envs))]
        self.distance_to_goal_z = [[] for _ in range(len(self.envs))]

        self.reached_goal = [False for _ in range(len(self.envs))]
        self.hard_reached_goal = [False for _ in range(len(self.envs))]

        # Log metric
        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)

        # Others
        self.apply_collision_force = True

        # SBC
        self.enable_sbc = enable_sbc
        self.sbc_neighbor_range = sbc_neighbor_range
        self.sbc_obst_range = sbc_obst_range
        self.no_sol_list = np.zeros(self.num_agents)

    def all_dynamics(self):
        return tuple(e.dynamics for e in self.envs)

    def get_rel_pos_vel_item(self, env_id, indices=None):
        i = env_id

        if indices is None:
            # if not specified explicitly, consider all neighbors
            indices = [j for j in range(self.num_agents) if j != i]

        cur_pos = self.pos[i]
        cur_vel = self.vel[i]
        pos_neighbor = np.stack([self.neighbor_pos_copy[j] for j in indices])
        vel_neighbor = np.stack([self.neighbor_vel_copy[j] for j in indices])
        pos_rel = pos_neighbor - cur_pos
        vel_rel = vel_neighbor - cur_vel
        return pos_rel, vel_rel

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        i = env_id
        pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])

        if self.neighbor_obs_type == 'pos':
            obs_neighbor_rel = np.array(pos_neighbors_rel)
        elif self.neighbor_obs_type == 'pos_vel':
            obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
        else:
            raise NotImplementedError("Incorrect neighbor_obs_type")

        return obs_neighbor_rel

    def extend_obs_space(self, obs, closest_drones):
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

                if self.neighbor_obs_type == 'pos':
                    new_rel_dist = np.array(rel_dist)
                elif self.neighbor_obs_type == 'pos_vel':
                    rel_pos_unit = rel_pos / rel_dist[:, None]
                    # new relative distance is a new metric that combines relative position and relative velocity
                    # the smaller the new_rel_dist, the closer the drones
                    new_rel_dist = rel_dist + np.sum(rel_pos_unit * rel_vel, axis=1)
                else:
                    raise RuntimeError("Incorrect neighbor_obs_type")

                rel_pos_index = new_rel_dist.argsort()
                rel_pos_index = rel_pos_index[:self.num_use_neighbor_obs]
                close_neighbor_indices.append(indices[i][rel_pos_index])

            return close_neighbor_indices
        else:
            raise RuntimeError("Incorrect number of neigbors")

    def add_neighborhood_obs(self, obs):
        indices = self.neighborhood_indices()
        obs_ext = self.extend_obs_space(obs, closest_drones=indices)
        return obs_ext

    def can_drones_fly(self):
        """
        Here we count the average number of collisions with the walls and ground in the last N episodes
        Returns: True if drones are considered proficient at flying
        """
        res = abs(np.mean(self.crashes_in_recent_episodes)) < 1 and len(self.crashes_in_recent_episodes) >= 10
        return res

    def calculate_room_collision(self):
        floor_collisions = np.array([env.dynamics.crashed_floor for env in self.envs])
        wall_collisions = np.array([env.dynamics.crashed_wall for env in self.envs])
        ceiling_collisions = np.array([env.dynamics.crashed_ceiling for env in self.envs])

        floor_crash_list = np.where(floor_collisions >= 1)[0]

        cur_wall_crash_list = np.where(wall_collisions >= 1)[0]
        wall_crash_list = np.setdiff1d(cur_wall_crash_list, self.prev_crashed_walls)

        cur_ceiling_crash_list = np.where(ceiling_collisions >= 1)[0]
        ceiling_crash_list = np.setdiff1d(cur_ceiling_crash_list, self.prev_crashed_ceiling)

        return floor_crash_list, wall_crash_list, ceiling_crash_list

    def init_scene_multi(self):
        models = tuple(e.dynamics.model for e in self.envs)
        for i in range(len(self.quads_view_mode)):
            self.scenes.append(Quadrotor3DSceneMulti(
                w=600, h=480, models=models, resizable=True, viewpoint=self.quads_view_mode[i],
                room_dims=self.room_dims, num_agents=self.num_agents, formation_size=self.quads_formation_size,
                render_speed=self.render_speed, vis_vel_arrows=False, vis_acc_arrows=True, viz_traces=50,
                viz_trace_nth_step=1, scene_index=i
            ))

    def reset(self):
        obs, rewards, dones, infos = [], [], [], []

        # Scenario reset
        if self.use_obstacles:
            transpose_obst_area_flag = np.random.choice([0, 1])
            self.obstacles.reset_randomization_and_get_obst_map(transpose_obst_area_flag=transpose_obst_area_flag)
            scenario_params = {
                'transpose_obst_area_flag': transpose_obst_area_flag,
                'obst_map': self.obstacles.obst_map,
                'cell_centers': self.obstacles.cell_centers,
            }
            self.scenario.reset(params=scenario_params)
        else:
            self.scenario.reset()

        # Replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_in_recent_episodes.append(self.crashes_last_episode)
            self.activate_replay_buffer = self.can_drones_fly()
            self.crashes_last_episode = 0

        for i, e in enumerate(self.envs):
            e.goal = self.scenario.goals[i]
            if self.scenario.spawn_points is None:
                e.spawn_point = self.scenario.goals[i]
            else:
                e.spawn_point = self.scenario.spawn_points[i]
            e.rew_coeff = self.rew_coeff

            observation = e.reset()
            obs.append(observation)
            self.pos[i, :] = e.dynamics.pos
            self.vel[i, :] = e.dynamics.vel

        # Neighbors
        self.neighbor_pos_copy = copy.deepcopy(self.pos)
        self.neighbor_vel_copy = copy.deepcopy(self.vel)

        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        # Obstacles
        if self.use_obstacles:
            quads_pos = np.array([e.dynamics.pos for e in self.envs])
            if self.obst_obs_type == "ToFs":
                quads_rots = np.array([e.dynamics.rot for e in self.envs])
                obs = self.obstacles.reset(obs=obs, quads_pos=quads_pos, quads_rots=quads_rots)
            else:
                obs = self.obstacles.reset(obs=obs, quads_pos=quads_pos)
            self.obst_quad_collisions_per_episode = self.obst_quad_collisions_after_settle = 0
            self.prev_obst_quad_collisions = []

        # Collision
        # # Collision: Neighbor
        self.collisions_per_episode = self.collisions_after_settle = self.collisions_final_5s = 0
        self.prev_drone_collisions = []

        # # Collision: Room
        self.collisions_room_per_episode = 0
        self.collisions_floor_per_episode = self.collisions_wall_per_episode = self.collisions_ceiling_per_episode = 0
        self.prev_crashed_walls = []
        self.prev_crashed_ceiling = []
        self.prev_crashed_room = []

        # Log
        # # Final Distance (1s / 3s / 5s)
        self.distance_to_goal = [[] for _ in range(len(self.envs))]
        self.distance_to_goal_xy = [[] for _ in range(len(self.envs))]
        self.distance_to_goal_z = [[] for _ in range(len(self.envs))]

        self.agent_col_agent = np.ones(self.num_agents)
        self.agent_col_obst = np.ones(self.num_agents)
        self.reached_goal = [False for _ in range(len(self.envs))]
        self.hard_reached_goal = [False for _ in range(len(self.envs))]
        self.low_h_count = 0
        self.no_sol_list = np.zeros(self.num_agents)

        # Rendering
        if self.quads_render:
            self.reset_scene = True
            self.quads_formation_size = self.scenario.formation_size
            self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}
        
        return obs

    def step(self, actions):
        obs, rewards, dones, infos = [], [], [], []

        for i, a in enumerate(actions):
            self.envs[i].rew_coeff = self.rew_coeff

            if self.enable_sbc:
                self_state = {
                    'position': self.envs[i].dynamics.pos,
                    'velocity': self.envs[i].dynamics.vel
                }
                neighbor_descriptions = []
                obstacle_descriptions = []

                # Add neighbor robot descriptions
                neighbor_distances = [np.linalg.norm(self_state['position'] - self.envs[j].dynamics.pos) for j in
                                      range(self.num_agents)]

                neighbor_ids = np.where(np.array(neighbor_distances) < self.sbc_neighbor_range)[0]
                for j in neighbor_ids:
                    if i == j:
                        continue

                    neighbor_descriptions.append({
                        'state': {
                            'position': self.envs[j].dynamics.pos,
                            'velocity': self.envs[j].dynamics.vel
                        },
                        'radius': self.envs[j].sbc_controller.sbc.radius,
                        'maximum_linf_acceleration_lower_bound': self.envs[j].sbc_controller.sbc.maximum_linf_acceleration,
                    })

                # Add obstacle descriptions
                self_pos = np.array([self_state['position'][0], self_state['position'][1]])
                obst_distances = [np.linalg.norm(np.array([obst_pos[0], obst_pos[1]]) - self_pos)
                                  for obst_pos in self.obstacles.obst_pos_arr]

                obst_ids = np.where(np.array(obst_distances) < self.sbc_obst_range)[0]
                for obst_id in obst_ids:
                    obst_pos = np.array(self.obstacles.obst_pos_arr[obst_id])[:2]
                    obstacle_descriptions.append({
                        'state': {
                            'position': obst_pos,
                            'velocity': np.zeros(2)
                        },
                        'radius': self.obstacles.obst_size_arr[obst_id] * 0.5,
                        'maximum_linf_acceleration_lower_bound': 0.0,
                    })


                sbc_data={
                    'self_state': self_state, 'neighbor_descriptions': neighbor_descriptions,
                    'obstacle_descriptions': obstacle_descriptions,
                }
            else:
                sbc_data = None


            observation, reward, done, info = self.envs[i].step(
                action=a,
                sbc_data=sbc_data,
            )
            obs.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

            self.no_sol_list[i] += float(info['no_sol_flag']) / self.envs[0].ep_len

            self.pos[i, :] = self.envs[i].dynamics.pos
        # 1. Calculate collisions: 1) between drones 2) with obstacles 3) with room
        # 1) Collisions between drones
        drone_col_matrix, curr_drone_collisions, distance_matrix = \
            calculate_collision_matrix(positions=self.pos, collision_threshold=self.collision_threshold)

        # # Filter curr_drone_collisions
        curr_drone_collisions = curr_drone_collisions.astype(int)
        curr_drone_collisions = np.delete(curr_drone_collisions, np.unique(
            np.where(curr_drone_collisions == [-1000, -1000])[0]), axis=0)

        old_quad_collision = set(map(tuple, self.prev_drone_collisions))
        new_quad_collision = np.array([x for x in curr_drone_collisions if tuple(x) not in old_quad_collision])

        self.last_step_unique_collisions = np.setdiff1d(curr_drone_collisions, self.prev_drone_collisions)

        # # Filter distance_matrix; Only contains quadrotor pairs with distance <= self.collision_threshold
        near_quad_ids = np.where(distance_matrix[:, 2] <= self.collision_falloff_threshold)
        distance_matrix = distance_matrix[near_quad_ids]

        # Collision between 2 drones counts as a single collision
        # # Calculate collisions (i) All collisions (ii) collisions after grace period
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        if collisions_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_after_settle += collisions_curr_tick
            for agent_id in self.last_step_unique_collisions:
                self.agent_col_agent[agent_id] = 0
        if collisions_curr_tick > 0 and self.envs[0].time_remain <= self.collisions_final_grace_period_steps:
            self.collisions_final_5s += collisions_curr_tick

        # # Aux: Neighbor Collisions
        self.prev_drone_collisions = curr_drone_collisions

        # 2) Collisions with obstacles
        if self.use_obstacles:
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            obst_quad_col_matrix, quad_obst_pair = self.obstacles.collision_detection(pos_quads=self.pos)
            # We assume drone can only collide with one obstacle at the same time.
            # Given this setting, in theory, the gap between obstacles should >= 0.1 (drone diameter: 0.46*2 = 0.92)
            self.curr_quad_col = np.setdiff1d(obst_quad_col_matrix, self.prev_obst_quad_collisions)
            collisions_obst_curr_tick = len(self.curr_quad_col)
            self.obst_quad_collisions_per_episode += collisions_obst_curr_tick

            if collisions_obst_curr_tick > 0 and self.envs[0].tick >= self.collisions_grace_period_steps:
                self.obst_quad_collisions_after_settle += collisions_obst_curr_tick
                for qid in self.curr_quad_col:
                    # Used for log agent_success
                    self.agent_col_obst[qid] = 0

            # # Aux: Obstacle Collisions
            self.prev_obst_quad_collisions = obst_quad_col_matrix

            if len(obst_quad_col_matrix) > 0:
                # We assign penalties to the drones which collide with the obstacles
                # And obst_quad_last_step_unique_collisions only include drones' id
                rew_obst_quad_collisions_raw[self.curr_quad_col] = -1.0

        # 3) Collisions with room
        floor_crash_list, wall_crash_list, ceiling_crash_list = self.calculate_room_collision()
        room_crash_list = np.unique(np.concatenate([floor_crash_list, wall_crash_list, ceiling_crash_list]))
        room_crash_list = np.setdiff1d(room_crash_list, self.prev_crashed_room)
        # # Aux: Room Collisions
        self.prev_crashed_walls = wall_crash_list
        self.prev_crashed_ceiling = ceiling_crash_list
        self.prev_crashed_room = room_crash_list

        # 2. Calculate rewards and infos for collision
        # 1) Between drones
        rew_collisions_raw = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        if len(distance_matrix) > 0:
            rew_proximity = -1.0 * calculate_drone_proximity_penalties(
                distance_matrix=distance_matrix, collision_falloff_threshold=self.collision_falloff_threshold,
                dt=self.control_dt, max_penalty=self.rew_coeff["quadcol_bin_smooth_max"], num_agents=self.num_agents,
            )
        else:
            rew_proximity = np.zeros(self.num_agents)

        # # Not in z overlap
        rew_z_overlap_raw = np.zeros(self.num_agents)
        pos_xy_list = np.array(self.pos)[:, :2]
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i == j:
                    continue
                if np.linalg.norm(pos_xy_list[i] - pos_xy_list[j]) < 2.0 * self.quad_arm:
                    rew_z_overlap_raw[i] = -1.0
                    rew_z_overlap_raw[j] = -1.0

        rew_z_overlap = 0.0 * rew_z_overlap_raw

        # 2) With obstacles
        rew_collisions_obst_quad = np.zeros(self.num_agents)
        if self.use_obstacles:
            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw

            # smooth penalty
            rew_obst_proximity = -1.0 * calculate_drone_obst_proximity_penalties(
                r_drone=self.quad_arm, obst_size_arr=self.obstacles.obst_size_arr,
                quads_pos=self.pos, obst_pos=self.obstacles.obst_pos_arr,
                penalty_coeff=self.rew_coeff["quads_obst_collision_prox_weight"],
                penalty_max=self.rew_coeff["quads_obst_collision_prox_max"],
                penalty_min=self.rew_coeff["quads_obst_collision_prox_min"],
                dt=self.control_dt
            )

        # 3) With room
        # # TODO: reward penalty
        if self.envs[0].tick >= self.collisions_grace_period_steps:
            self.collisions_room_per_episode += len(room_crash_list)
            self.collisions_floor_per_episode += len(floor_crash_list)
            self.collisions_wall_per_episode += len(wall_crash_list)
            self.collisions_ceiling_per_episode += len(ceiling_crash_list)

        # Reward & Info
        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]
            rewards[i] += rew_proximity[i]
            # Z overlap
            rewards[i] += rew_z_overlap[i]

            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]
            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]
            # Z overlap
            infos[i]["rewards"]["rewraw_z_overlap"] = rew_z_overlap_raw[i]
            infos[i]["rewards"]["rew_z_overlap"] = rew_z_overlap[i]

            if self.use_obstacles:
                rewards[i] += rew_collisions_obst_quad[i]
                rewards[i] += rew_obst_proximity[i]

                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]
                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rew_obst_proximity"] = rew_obst_proximity[i]

            self.distance_to_goal[i].append(np.linalg.norm(self.envs[i].dynamics.pos - self.envs[i].goal[:3]))
            self.distance_to_goal_xy[i].append(np.linalg.norm(self.envs[i].dynamics.pos[:2] - self.envs[i].goal[:2]))
            self.distance_to_goal_z[i].append(self.envs[i].dynamics.pos[2] - self.envs[i].goal[:3][2])

            if self.envs[0].time_remain <= 200:
                reach_len_bool = len(self.distance_to_goal[i]) >= 5
                reach_goal_bool = np.mean(self.distance_to_goal[i][-5:]) < self.scenario.approch_goal_metric

                hard_reach_len_bool = len(self.distance_to_goal[i]) >= 100
                hard_reach_goal_bool = np.mean(self.distance_to_goal[i][-100:]) < self.scenario.approch_goal_metric

                if not self.reached_goal[i] and reach_len_bool and reach_goal_bool:
                    self.reached_goal[i] = True

                if not self.hard_reached_goal[i] and hard_reach_len_bool and hard_reach_goal_bool:
                    self.hard_reached_goal[i] = True

        # 3. Applying random forces: 1) aerodynamics 2) between drones 3) obstacles 4) room
        self_state_update_flag = False

        # # 1) aerodynamics
        if self.use_downwash:
            envs_dynamics = [env.dynamics for env in self.envs]
            applied_downwash_list = perform_downwash(drones_dyn=envs_dynamics, dt=self.control_dt)
            downwash_agents_list = np.where(applied_downwash_list == 1)[0]
            if len(downwash_agents_list) > 0:
                self_state_update_flag = True

        # # 2) Drones
        if self.apply_collision_force:
            if len(new_quad_collision) > 0:
                self_state_update_flag = True
                for val in new_quad_collision:
                    dyn1, dyn2 = self.envs[val[0]].dynamics, self.envs[val[1]].dynamics
                    dyn1.vel, dyn1.omega, dyn2.vel, dyn2.omega = perform_collision_between_drones(
                        pos1=dyn1.pos, vel1=dyn1.vel, omega1=dyn1.omega, pos2=dyn2.pos, vel2=dyn2.vel, omega2=dyn2.omega)

            # # 3) Obstacles
            if self.use_obstacles:
                if len(self.curr_quad_col) > 0:
                    self_state_update_flag = True
                    for val in self.curr_quad_col:
                        obstacle_id = quad_obst_pair[int(val)]
                        obstacle_pos = self.obstacles.obst_pos_arr[int(obstacle_id)]
                        obstacle_size = self.obstacles.obst_size_arr[int(obstacle_id)]
                        perform_collision_with_obstacle(
                            drone_dyn=self.envs[int(val)].dynamics, obstacle_pos=obstacle_pos,
                            obstacle_size=obstacle_size
                        )

            # # 4) Room
            if len(wall_crash_list) > 0 or len(ceiling_crash_list) > 0:
                self_state_update_flag = True

                for val in wall_crash_list:
                    perform_collision_with_wall(drone_dyn=self.envs[val].dynamics, room_box=self.envs[0].room_box)

                for val in ceiling_crash_list:
                    perform_collision_with_ceiling(drone_dyn=self.envs[val].dynamics)

        # 4. Run the scenario passed to self.quads_mode
        self.scenario.step()

        # 5. Collect final observations
        # Collect positions after physical interaction
        for i in range(self.num_agents):
            self.pos[i, :] = self.envs[i].dynamics.pos
            self.vel[i, :] = self.envs[i].dynamics.vel

        if self_state_update_flag:
            obs = [e.state_vector(e) for e in self.envs]

        # Concatenate observations of neighbor drones
        # Update neighbor obs
        if self.envs[0].tick % int(100 // self.neighbor_obs_update_freq) == 0:
            self.neighbor_pos_copy = copy.deepcopy(self.pos)
            self.neighbor_vel_copy = copy.deepcopy(self.vel)

        if self.num_use_neighbor_obs > 0:
            obs = self.add_neighborhood_obs(obs)

        # Concatenate obstacle observations
        if self.use_obstacles:
            if self.obst_obs_type == "ToFs":
                quads_rots = np.array([e.dynamics.rot for e in self.envs])
                obs = self.obstacles.step(obs=obs, quads_pos=self.pos, quads_rots=quads_rots)
            else:
                obs = self.obstacles.step(obs=obs, quads_pos=self.pos)

        # 6. Update info for replay buffer
        # Once agent learns how to take off, activate the replay buffer
        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        # Rendering
        if self.quads_render:
            # Collisions with room
            ground_collisions = [1.0 if env.dynamics.on_floor else 0.0 for env in self.envs]
            if self.use_obstacles:
                obst_coll = [1.0 if i < 0 else 0.0 for i in rew_obst_quad_collisions_raw]
            else:
                obst_coll = [0.0 for _ in range(self.num_agents)]
            self.all_collisions = {'drone': drone_col_matrix, 'ground': ground_collisions,
                                   'obstacle': obst_coll}

        # Low height
        if self.envs[0].tick >= self.collisions_grace_period_steps:
            for i in range(self.num_agents):
                if self.envs[i].dynamics.pos[2] < 0.2:
                    self.low_h_count += 1

        # 7. DONES
        if any(dones):
            scenario_name = self.scenario.name()[9:]
            if self.scenario.goal_scenario_flag == 0:
                if self.scenario.in_obst_area == 0:
                    scenario_name += '_diff_out_obst'
                else:
                    scenario_name += '_diff_in_obst'
            else:
                if self.scenario.in_obst_area == 0:
                    scenario_name += '_same_out_obst'
                else:
                    scenario_name += '_same_in_obst'

            for i in range(len(infos)):
                if self.saved_in_replay_buffer:
                    infos[i]['episode_extra_stats'] = {
                        f'replay/{scenario_name}/num_collisions': self.collisions_per_episode,
                        f'replay/{scenario_name}/num_collisions_obst': self.obst_quad_collisions_per_episode,
                    }
                else:
                    self.distance_to_goal = np.array(self.distance_to_goal)
                    self.distance_to_goal_xy = np.array(self.distance_to_goal_xy)
                    self.distance_to_goal_z = np.array(self.distance_to_goal_z)

                    self.reached_goal = np.array(self.reached_goal)
                    self.hard_reached_goal = np.array(self.hard_reached_goal)
                    infos[i]['episode_extra_stats'] = {
                        'num_collisions': self.collisions_per_episode,
                        'num_collisions_with_room': self.collisions_room_per_episode,
                        'num_collisions_with_floor': self.collisions_floor_per_episode,
                        'num_collisions_with_wall': self.collisions_wall_per_episode,
                        'num_collisions_with_ceiling': self.collisions_ceiling_per_episode,
                        'num_collisions_after_settle': self.collisions_after_settle,
                        f'{scenario_name}/num_collisions': self.collisions_after_settle,
                        f'{scenario_name}/z_overlap': rew_z_overlap_raw[i],

                        'num_collisions_final_5_s': self.collisions_final_5s,
                        f'{scenario_name}/num_collisions_final_5_s': self.collisions_final_5s,

                        'num_low_h': self.low_h_count / self.num_agents,
                        f'{scenario_name}/num_low_h': self.low_h_count / self.num_agents,

                        f'{scenario_name}/distance_to_goal_1s': np.mean(self.distance_to_goal[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/{self.obstacles.obst_density}/distance_to_goal_1s': np.mean(self.distance_to_goal[i, int(-1 * self.control_freq):]),


                        f'{scenario_name}/distance_to_goal_3s': np.mean(self.distance_to_goal[i, int(-3 * self.control_freq):]),
                        f'{scenario_name}/distance_to_goal_5s': np.mean(self.distance_to_goal[i, int(-5 * self.control_freq):]),

                        f'{scenario_name}/xy_distance_to_goal_1s': np.mean(self.distance_to_goal_xy[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/{self.obstacles.obst_density}/xy_distance_to_goal_1s': np.mean(
                            self.distance_to_goal_xy[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/z_distance_to_goal_1s': np.mean(self.distance_to_goal_z[i, int(-1 * self.control_freq):]),
                        f'{scenario_name}/{self.obstacles.obst_density}/z_distance_to_goal_1s': np.mean(self.distance_to_goal_z[i, int(-1 * self.control_freq):]),
                    }

                    if self.use_obstacles:
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = self.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad_after_settle'] = self.obst_quad_collisions_after_settle
                        infos[i]['episode_extra_stats'][f'{scenario_name}/num_collisions_obst'] = self.obst_quad_collisions_per_episode

            if not self.saved_in_replay_buffer:
                # agent_success_rate: base_success_rate, based on per agent
                # 0: collision; 1: no collision
                agent_col_flag_list = np.logical_and(self.agent_col_agent, self.agent_col_obst)
                agent_success_flag_list = np.logical_and(agent_col_flag_list, self.reached_goal)
                agent_success_ratio = 1.0 * np.sum(agent_success_flag_list) / self.num_agents

                agent_hard_success_flag_list = np.logical_and(agent_col_flag_list, self.hard_reached_goal)
                agent_hard_success_ratio = 1.0 * np.sum(agent_hard_success_flag_list) / self.num_agents

                # agent_deadlock_rate
                # Doesn't approach to the goal while no collisions with other objects
                agent_deadlock_list = np.logical_and(agent_col_flag_list, 1 - self.reached_goal)
                agent_deadlock_ratio = 1.0 * np.sum(agent_deadlock_list) / self.num_agents

                # agent_col_rate
                # Collide with other drones and obstacles
                agent_col_ratio = 1.0 - np.sum(agent_col_flag_list) / self.num_agents

                # agent_neighbor_col_rate
                agent_neighbor_col_ratio = 1.0 - np.sum(self.agent_col_agent) / self.num_agents
                # agent_obst_col_rate
                agent_obst_col_ratio = 1.0 - np.sum(self.agent_col_obst) / self.num_agents

                for i in range(len(infos)):
                    # agent_success_rate
                    infos[i]['episode_extra_stats']['metric/agent_success_rate'] = agent_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_success_rate'] = agent_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/{self.obstacles.obst_density}/agent_success_rate'] = agent_success_ratio

                    infos[i]['episode_extra_stats']['metric/agent_hard_success_rate'] = agent_hard_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_hard_success_rate'] = agent_hard_success_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/{self.obstacles.obst_density}/agent_hard_success_rate'] = agent_hard_success_ratio

                    # agent_deadlock_rate
                    infos[i]['episode_extra_stats']['metric/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_deadlock_rate'] = agent_deadlock_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/{self.obstacles.obst_density}/agent_deadlock_rate'] = agent_deadlock_ratio
                    # agent_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_col_rate'] = agent_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_col_rate'] = agent_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/{self.obstacles.obst_density}/agent_col_rate'] = agent_col_ratio

                    # agent_neighbor_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/{self.obstacles.obst_density}/agent_neighbor_col_rate'] = agent_neighbor_col_ratio
                    # agent_obst_col_rate
                    infos[i]['episode_extra_stats']['metric/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/agent_obst_col_rate'] = agent_obst_col_ratio
                    infos[i]['episode_extra_stats'][f'{scenario_name}/{self.obstacles.obst_density}/agent_obst_col_rate'] = agent_obst_col_ratio

                    infos[i]['episode_extra_stats']['metric/sbc_no_sol_rate'] = self.no_sol_list[i]
                    infos[i]['episode_extra_stats'][f'{scenario_name}/sbc_no_sol_rate'] = self.no_sol_list[i]
                    infos[i]['episode_extra_stats'][f'{scenario_name}/{self.obstacles.obst_density}/sbc_no_sol_rate'] = self.no_sol_list[i]

                    infos[i]['episode_extra_stats']['metric/distance_to_goal_1s'] = np.mean(self.distance_to_goal[i, int(-1 * self.control_freq):])
                    infos[i]['episode_extra_stats']['metric/distance_to_goal_3s'] = np.mean(self.distance_to_goal[i, int(-3 * self.control_freq):])
                    infos[i]['episode_extra_stats']['metric/distance_to_goal_5s'] = np.mean(self.distance_to_goal[i, int(-5 * self.control_freq):])
                    infos[i]['episode_extra_stats']['metric/xy_distance_to_goal_1s'] = np.mean(self.distance_to_goal_xy[i, int(-1 * self.control_freq):])
                    infos[i]['episode_extra_stats']['metric/z_distance_to_goal_1s'] = np.mean(self.distance_to_goal_z[i, int(-1 * self.control_freq):])

            obs = self.reset()
            # terminate the episode for all "sub-envs"
            dones = [True] * len(dones)

        return obs, rewards, dones, infos

    def render(self, verbose=False):
        models = tuple(e.dynamics.model for e in self.envs)

        if len(self.scenes) == 0:
            from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
            self.init_scene_multi()

        if self.reset_scene:
            for i in range(len(self.scenes)):
                self.scenes[i].update_models(models=models)
                self.scenes[i].formation_size = self.quads_formation_size
                self.scenes[i].update_env(room_dims=self.room_dims, obstacles=self.obstacles)

                self.scenes[i].reset(
                    goals=tuple(e.goal for e in self.envs), dynamics=self.all_dynamics(),
                    obstacles=self.obstacles, collisions=self.all_collisions
                )

            self.reset_scene = False

        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.scenario.formation_size
        else:
            for i in range(len(self.scenes)):
                self.scenes[i].formation_size = self.scenario.formation_size
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
        frames = []
        first_spawn = None
        for i in range(len(self.scenes)):
            frame, first_spawn = self.scenes[i].render_chase(
                all_dynamics=self.all_dynamics(), goals=goals, collisions=self.all_collisions, mode=self.render_mode,
                obstacles=self.obstacles, first_spawn=first_spawn
            )
            frames.append(frame)
        # Update the formation size of the scenario
        if self.quads_mode == "mix":
            for i in range(len(self.scenes)):
                self.scenario.scenario.update_formation_size(self.scenes[i].formation_size)
        else:
            for i in range(len(self.scenes)):
                self.scenario.update_formation_size(self.scenes[i].formation_size)

        render_time = time.time() - render_start

        desired_time_between_frames = realtime_control_period * self.frames_since_last_render / self.render_speed
        time_to_sleep = desired_time_between_frames - simulation_time - render_time

        # wait so we don't simulate/render faster than realtime
        if self.render_mode == "human" and time_to_sleep > 0:
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

        if self.render_mode == "rgb_array":
            return frame

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

        # warning! deep-copied env has its scene uninitialized! We need to reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
