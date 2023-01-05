import copy
from collections import deque

import numpy as np
import time
import gym

from copy import deepcopy

from gym_art.quadrotor_multi.quad_utils import perform_collision_between_drones, perform_collision_with_obstacle, \
    calculate_collision_matrix, calculate_drone_proximity_penalties, calculate_obst_drone_proximity_penalties

from gym_art.quadrotor_multi.quadrotor_multi_obstacles import MultiObstacles
from gym_art.quadrotor_multi.quadrotor_single import GRAV, QuadrotorSingle
from gym_art.quadrotor_multi.quadrotor_multi_visualization import Quadrotor3DSceneMulti
from gym_art.quadrotor_multi.quad_scenarios import create_scenario
from gym_art.quadrotor_multi.quad_obstacle_utils import OBSTACLES_SHAPE_LIST

EPS = 1E-6


class QuadrotorEnvMulti(gym.Env):
    def __init__(self,
                 num_agents,
                 dynamics_params='DefaultQuad', dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2, obs_repr='xyz_vxyz_R_omega', ep_time=7, obstacles_num=0, room_length=10, room_width=10,
                 room_height=10, init_random_state=False, rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV,
                 resample_goals=False, t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False,
                 quads_mode='static_same_goal', quads_formation='circle_horizontal', quads_formation_size=-1.0,
                 swarm_obs='none', quads_use_numba=False, quads_settle=False, quads_settle_range_meters=1.0,
                 quads_vel_reward_out_range=0.8, quads_obstacle_mode='no_obstacles', quads_view_mode='local',
                 quads_obstacle_num=0, quads_obstacle_type='sphere', quads_obstacle_size=0.0, collision_force=True,
                 adaptive_env=False, obstacle_traj='gravity', local_obs=-1, collision_hitbox_radius=2.0,
                 collision_falloff_radius=2.0, collision_smooth_max_penalty=10.0,
                 local_metric='dist', local_coeff=0.0, use_replay_buffer=False,
                 obstacle_obs_mode='relative', obst_penalty_fall_off=10.0, vis_acc_arrows=False,
                 viz_traces=25, viz_trace_nth_step=1):

        super().__init__()

        self.num_agents = num_agents
        self.swarm_obs = swarm_obs
        assert local_obs <= self.num_agents - 1 or local_obs == -1, f'Invalid value ({local_obs}) passed to --local_obs. Should be 0 < n < num_agents - 1, or -1'
        if local_obs == -1:
            self.num_use_neighbor_obs = self.num_agents - 1
        else:
            self.num_use_neighbor_obs = local_obs

        self.local_metric = local_metric
        self.local_coeff = local_coeff
        # Set to True means that sample_factory will treat it as a multi-agent vectorized environment even with
        # num_agents=1. More info, please look at sample-factory: envs/quadrotors/wrappers/reward_shaping.py
        self.is_multiagent = True
        self.room_dims = (room_length, room_width, room_height)

        self.envs = []
        self.adaptive_env = adaptive_env
        self.quads_view_mode = quads_view_mode

        for i in range(self.num_agents):
            e = QuadrotorSingle(
                dynamics_params, dynamics_change, dynamics_randomize_every, dyn_sampler_1, dyn_sampler_2,
                raw_control, raw_control_zero_middle, dim_mode, tf_control, sim_freq, sim_steps,
                obs_repr, ep_time, obstacles_num, room_length, room_width, room_height, init_random_state,
                rew_coeff, sense_noise, verbose, gravity, t2w_std, t2t_std, excite, dynamics_simplification,
                quads_use_numba, self.swarm_obs, self.num_agents, quads_settle, quads_settle_range_meters,
                quads_vel_reward_out_range, quads_view_mode, quads_obstacle_mode, quads_obstacle_num,
                self.num_use_neighbor_obs
            )
            self.envs.append(e)

        self.resample_goals = resample_goals

        # we don't actually create a scene object unless we want to render stuff
        self.scene = None

        self.action_space = self.envs[0].action_space
        self.observation_space = self.envs[0].observation_space

        # reward shaping
        self.rew_coeff = dict(
            pos=1., effort=0.05, action_change=0., crash=1., orient=1., yaw=0., rot=0., attitude=0., spin=0.1, vel=0.,
            quadcol_bin=0., quadcol_bin_smooth_max=0.,
            quadsettle=0., quadcol_bin_obst=0., quadcol_bin_obst_smooth_max=0.0
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

        # Aux variables for observation space of quads
        self.quad_arm = self.envs[0].dynamics.arm
        self.control_freq = self.envs[0].control_freq
        self.control_dt = 1.0 / self.control_freq
        self.pos = np.zeros([self.num_agents, 3])  # Matrix containing all positions
        self.quads_mode = quads_mode
        if obs_repr == 'xyz_vxyz_R_omega':
            obs_self_size = 18
        elif obs_repr == 'xyz_vxyz_R_omega_wall':
            obs_self_size = 24
        else:
            raise NotImplementedError(f'{obs_repr} not supported!')

        if self.swarm_obs == 'pos_vel':
            self.neighbor_obs_size = 6
        elif self.swarm_obs == 'pos_vel_goals_ndist_gdist':
            self.neighbor_obs_size = 11
        elif self.swarm_obs == 'pos_vel_goals':
            self.neighbor_obs_size = 9
        elif self.swarm_obs == 'none':
            self.neighbor_obs_size = 0
        else:
            raise NotImplementedError(f'Unknown value {self.swarm_obs} passed to --neighbor_obs_type')
        self.clip_neighbor_space_length = self.num_use_neighbor_obs * self.neighbor_obs_size
        self.clip_neighbor_space_min_box = self.observation_space.low[obs_self_size:obs_self_size+self.clip_neighbor_space_length]
        self.clip_neighbor_space_max_box = self.observation_space.high[obs_self_size:obs_self_size+self.clip_neighbor_space_length]

        # Aux variables for rewards
        self.rews_settle = np.zeros(self.num_agents)
        self.rews_settle_raw = np.zeros(self.num_agents)

        # Aux variables for scenarios
        self.scenario = create_scenario(quads_mode=quads_mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims, room_dims_callback=self.set_room_dims, rew_coeff=self.rew_coeff,
                                        quads_formation=quads_formation, quads_formation_size=quads_formation_size)
        self.quads_formation_size = quads_formation_size
        self.goal_central = np.array([0., 0., 2.])

        # Set Obstacles
        self.multi_obstacles = None
        self.obstacle_mode = quads_obstacle_mode
        self.obstacle_num = quads_obstacle_num
        self.use_obstacles = self.obstacle_mode != 'no_obstacles' and self.obstacle_num > 0
        if self.use_obstacles:
            obstacle_max_init_vel = 4.0 * self.envs[0].max_init_vel
            obstacle_init_box = self.envs[0].box  # box of env is: 2 meters
            # This parameter is used to judge whether obstacles are out of room, and then, we can reset the obstacles
            self.obstacle_room = self.envs[0].room_box  # [[-5, -5, 0], [5, 5, 10]]
            dt = 1.0 / sim_freq
            self.set_obstacles = np.zeros(self.obstacle_num, dtype=bool)
            self.obstacle_shape = quads_obstacle_type
            self.obst_penalty_fall_off = obst_penalty_fall_off
            self.multi_obstacles = MultiObstacles(
                mode=self.obstacle_mode, num_obstacles=self.obstacle_num, max_init_vel=obstacle_max_init_vel,
                init_box=obstacle_init_box, dt=dt, quad_size=self.quad_arm, shape=self.obstacle_shape,
                size=quads_obstacle_size, traj=obstacle_traj, obs_mode=obstacle_obs_mode
            )

            # collisions between obstacles and quadrotors
            self.obst_quad_collisions_per_episode = 0
            self.prev_obst_quad_collisions = []

        # set render
        self.simulation_start_time = 0
        self.frames_since_last_render = self.render_skip_frames = 0
        self.render_every_nth_frame = 1
        self.render_speed = 1.0  # set to below 1 slowmo, higher than 1 for fast forward (if simulator can keep up)

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

        # set to true whenever we need to reset the OpenGL scene in render()
        self.reset_scene = False
        self.vis_acc_arrows = vis_acc_arrows
        self.viz_traces = viz_traces
        self.viz_trace_nth_step = viz_trace_nth_step

        self.use_replay_buffer = use_replay_buffer
        self.activate_replay_buffer = False  # only start using the buffer after the drones learn how to fly
        self.saved_in_replay_buffer = False  # since the same collisions happen during replay, we don't want to keep resaving the same event
        self.last_step_unique_collisions = False
        self.crashes_in_recent_episodes = deque([], maxlen=100)
        self.crashes_last_episode = 0

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
        pos_rel = pos_neighbor - cur_pos
        vel_rel = vel_neighbor - cur_vel
        return pos_rel, vel_rel

    def get_rel_pos_vel_stack(self):
        rel_pos_stack, rel_vel_stack = [], []
        for i in range(self.num_agents):
            pos_rel, vel_rel = self.get_rel_pos_vel_item(env_id=i)
            rel_pos_stack.append(pos_rel)
            rel_vel_stack.append(vel_rel)
        return np.array(rel_pos_stack), np.array(rel_vel_stack)

    def get_obs_neighbor_rel(self, env_id, closest_drones):
        i = env_id
        pos_neighbors_rel, vel_neighbors_rel = self.get_rel_pos_vel_item(env_id=i, indices=closest_drones[i])

        if self.swarm_obs == 'pos_vel':
            obs_neighbor_rel = np.concatenate((pos_neighbors_rel, vel_neighbors_rel), axis=1)
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
        assert self.swarm_obs in ['pos_vel', 'pos_vel_goals', 'pos_vel_goals_ndist_gdist'], \
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
        )

    def reset(self):
        obs, rewards, dones, infos = [], [], [], []
        self.scenario.reset()
        self.quads_formation_size = self.scenario.formation_size
        self.goal_central = np.mean(self.scenario.goals, axis=0)

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

            observation = e.reset()
            obs.append(observation)

        # extend obs to see neighbors
        obs = self.add_neighborhood_obs(obs)

        # Reset Obstacles
        if self.use_obstacles:
            self.set_obstacles = np.zeros(self.obstacle_num, dtype=bool)
            quads_pos = np.array([e.dynamics.pos for e in self.envs])
            quads_vel = np.array([e.dynamics.vel for e in self.envs])
            obs = self.multi_obstacles.reset(obs=obs, quads_pos=quads_pos, quads_vel=quads_vel,
                                             set_obstacles=self.set_obstacles, formation_size=self.quads_formation_size,
                                             goal_central=self.goal_central)
            self.obst_quad_collisions_per_episode = 0
            self.prev_obst_quad_collisions = []

        self.all_collisions = {val: [0.0 for _ in range(len(self.envs))] for val in ['drone', 'ground', 'obstacle']}

        self.collisions_per_episode = self.collisions_after_settle = 0
        self.prev_drone_collisions, self.curr_drone_collisions = [], []

        self.reset_scene = True
        self.crashes_last_episode = 0
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

        obs = self.add_neighborhood_obs(obs)

        if self.use_replay_buffer and not self.activate_replay_buffer:
            self.crashes_last_episode += infos[0]["rewards"]["rew_crash"]

        # Calculating collisions between drones
        drone_col_matrix, self.curr_drone_collisions, distance_matrix = calculate_collision_matrix(self.pos, self.quad_arm, self.collision_hitbox_radius)

        self.last_step_unique_collisions = np.setdiff1d(self.curr_drone_collisions, self.prev_drone_collisions)

        # collision between 2 drones counts as a single collision
        collisions_curr_tick = len(self.last_step_unique_collisions) // 2
        self.collisions_per_episode += collisions_curr_tick

        if collisions_curr_tick > 0:
            if self.envs[0].tick >= self.collisions_grace_period_seconds * self.control_freq:
                self.collisions_after_settle += collisions_curr_tick

        self.prev_drone_collisions = self.curr_drone_collisions

        rew_collisions_raw = np.zeros(self.num_agents)
        if self.last_step_unique_collisions.any():
            rew_collisions_raw[self.last_step_unique_collisions] = -1.0
        rew_collisions = self.rew_coeff["quadcol_bin"] * rew_collisions_raw

        # penalties for being too close to other drones
        rew_proximity = -1.0 * calculate_drone_proximity_penalties(
            distance_matrix=distance_matrix, arm=self.quad_arm, dt=self.control_dt,
            penalty_fall_off=self.collision_falloff_radius,
            max_penalty=self.rew_coeff["quadcol_bin_smooth_max"],
            num_agents=self.num_agents,
        )

        # COLLISION BETWEEN QUAD AND OBSTACLE(S)
        if self.use_obstacles:
            obst_quad_col_matrix, curr_obst_quad_collisions, curr_all_collisions, obst_quad_distance_matrix \
                = self.multi_obstacles.collision_detection(pos_quads=self.pos, set_obstacles=self.set_obstacles)
            obst_quad_last_step_unique_collisions = np.setdiff1d(curr_obst_quad_collisions, self.prev_obst_quad_collisions)
            self.obst_quad_collisions_per_episode += len(obst_quad_last_step_unique_collisions)
            self.prev_obst_quad_collisions = curr_obst_quad_collisions

            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            if obst_quad_last_step_unique_collisions.any():
                # We assign penalties to the drones which collide with the obstacles
                # And obst_quad_last_step_unique_collisions only include drones' id
                rew_obst_quad_collisions_raw[obst_quad_last_step_unique_collisions] = -1.0

            rew_collisions_obst_quad = self.rew_coeff["quadcol_bin_obst"] * rew_obst_quad_collisions_raw

            # penalties for low distance between obstacles and drones
            obstacles_radius = np.stack([self.multi_obstacles.obstacles[i].size / 2 for i in range(self.obstacle_num)])
            rew_obst_quad_proximity = -1.0 * calculate_obst_drone_proximity_penalties(
                distance_matrix=obst_quad_distance_matrix, arm=self.quad_arm, dt=self.control_dt,
                penalty_fall_off=self.obst_penalty_fall_off,
                max_penalty=self.rew_coeff["quadcol_bin_obst_smooth_max"],
                num_agents=self.num_agents,
                obstacles_radius=obstacles_radius
            )
        else:
            obst_quad_col_matrix = np.zeros((self.num_agents, self.obstacle_num))
            curr_all_collisions = []
            rew_obst_quad_collisions_raw = np.zeros(self.num_agents)
            rew_collisions_obst_quad = np.zeros(self.num_agents)
            rew_obst_quad_proximity = np.zeros(self.num_agents)

        # Collisions with ground
        ground_collisions = [1.0 if pos[2] < 0.25 else 0.0 for pos in self.pos]

        self.all_collisions = {'drone': np.sum(drone_col_matrix, axis=1), 'ground': ground_collisions,
                               'obstacle': np.sum(obst_quad_col_matrix, axis=1)}

        # Applying random forces for all collisions between drones and obstacles
        if self.apply_collision_force:
            for val in self.curr_drone_collisions:
                perform_collision_between_drones(self.envs[val[0]].dynamics, self.envs[val[1]].dynamics)
            for val in curr_all_collisions:
                perform_collision_with_obstacle(
                    drone_dyn=self.envs[val[0]].dynamics, obstacle_dyn=self.multi_obstacles.obstacles[val[1]],
                    quad_arm=self.quad_arm)

        for i in range(self.num_agents):
            rewards[i] += rew_collisions[i]
            infos[i]["rewards"]["rew_quadcol"] = rew_collisions[i]
            infos[i]["rewards"]["rewraw_quadcol"] = rew_collisions_raw[i]

            rewards[i] += rew_proximity[i]
            infos[i]["rewards"]["rew_proximity"] = rew_proximity[i]

            if self.use_obstacles:
                rewards[i] += rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rew_quadcol_obstacle"] = rew_collisions_obst_quad[i]
                infos[i]["rewards"]["rewraw_quadcol_obstacle"] = rew_obst_quad_collisions_raw[i]

                rewards[i] += rew_obst_quad_proximity[i]
                infos[i]["rewards"]["rew_obst_quad_proximity"] = rew_obst_quad_proximity[i]

        # run the scenario passed to self.quads_mode
        infos, rewards = self.scenario.step(infos=infos, rewards=rewards, pos=self.pos)

        # For obstacles
        quads_vel = np.array([e.dynamics.vel for e in self.envs])

        if self.obstacle_mode == 'dynamic' and self.obstacle_num > 0:
            tmp_obs = self.multi_obstacles.step(obs=obs, quads_pos=self.pos, quads_vel=quads_vel,
                                                set_obstacles=self.set_obstacles)

            # If there are still at least one obstacle flying in the air, we should check the function below
            # and reset the counter only if all obstacles hit the floor
            if self.set_obstacles.any():
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

        # DONES
        if any(dones):
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
                        infos[i]['episode_extra_stats']['num_collisions_obst_quad'] = self.obst_quad_collisions_per_episode
                        infos[i]['episode_extra_stats'][f'num_collisions_obst_{self.scenario.name()}'] = self.obst_quad_collisions_per_episode

            obs = self.reset()
            dones = [True] * len(dones)  # terminate the episode for all "sub-envs"

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
        frame = self.scene.render_chase(all_dynamics=self.all_dynamics(), goals=goals, collisions=self.all_collisions,
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

        if mode == "rgb_array":
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

        # warning! deep-copied env has its scene uninitialized! We gotta reuse one from the existing env
        # to avoid creating tons of windows
        copied_env.scene = None

        return copied_env
