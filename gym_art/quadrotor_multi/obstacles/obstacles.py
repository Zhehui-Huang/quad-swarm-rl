import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_ToFs_depthmap, \
    obst_generation_given_density


class MultiObstacles:
    def __init__(self, params):
        # Initialize parameters
        # # Quadrotor parameters
        self.quad_radius = params['quad_radius']

        # # Obstacle parameters
        self.obst_obs_type = params['obst_obs_type']
        self.obst_noise = params['obst_noise']
        self.num_rays = params['obst_tof_resolution']

        self.obst_grid_size = params['obst_grid_size']
        self.obst_grid_size_random = params['obst_grid_size_random']
        self.obst_grid_size_range = params['obst_grid_size_range']

        self.obst_density_random = params['obst_density_random']
        self.obst_density_min = params['obst_density_min']
        self.obst_density_max = params['obst_density_max']
        self.obst_density = params['obst_density']
        self.obst_num = 0

        self.obst_size_random = params['obst_size_random']
        self.obst_min_gap_threshold = params['obst_min_gap_threshold']
        self.obst_size_min = params['obst_size_min']
        self.obst_size_max = params['obst_size_max']
        self.obst_size = params['obst_size']
        self.obst_size_range = [self.obst_size_min, self.obst_size_max]
        self.obst_size_arr = []

        # Aux
        obst_critic_obs = params['obst_critic_obs']
        critic_rnn_size = params['critic_rnn_size']

        self.obst_spawn_area = params['obst_spawn_area']
        self.obst_spawn_center = params['obst_spawn_center']
        self.room_dims = params['room_dims']
        self.sim2real_scenario = params['sim2real_scenario']

        self.obst_map = None
        self.obst_pos_arr = None
        self.cell_centers = None

        self.resolution = 0.1
        self.fov_angle = 45 * np.pi / 180 
        self.scan_angle_arr = np.array([0., np.pi/2, np.pi, -np.pi/2])
        self.angle_noise_std = 0.1
        self.prev = None
        self.tick = 0
        self.range_max = 2.0
        if self.num_rays == 4:
            self.sample_freq = 3
        else:
            self.sample_freq = 7 # 14 Hz, conservative but better than 6 being 16hz.

        self.use_obst_octomap_critic = False
        obst_obs_diff_ac = (self.obst_obs_type != obst_critic_obs)
        if critic_rnn_size > 0 and obst_obs_diff_ac:
            if obst_critic_obs == 'octomap':
                self.use_obst_octomap_critic = True

    def reset_randomization_and_get_obst_map(self, transpose_obst_area_flag):
        if self.obst_grid_size_random:
            tmp_obst_grid_size = np.random.uniform(
                low=self.obst_grid_size_range[0] - 0.049,
                high=self.obst_grid_size_range[1] + 0.049
            )
            self.obst_grid_size = np.round(tmp_obst_grid_size, decimals=1)

        if self.obst_density_random:
            self.obst_density = round(
                np.random.choice(np.arange(self.obst_density_min, self.obst_density_max + 0.01, 0.1)), 1
            )

        if self.obst_size_random:
            tmp_obst_size_max = self.obst_grid_size - self.obst_min_gap_threshold
            if tmp_obst_size_max < self.obst_size_min:
                raise ValueError(f"Obstacle size: {tmp_obst_size_max} is too small for the minimum gap threshold: {self.obst_min_gap_threshold}")

            tmp_obst_max = min(tmp_obst_size_max, self.obst_size_max)
            self.obst_size_range = [self.obst_size_min, tmp_obst_max]
        else:
            self.obst_size_range = [self.obst_size, self.obst_size]

        if self.sim2real_scenario is not None:
            self.obst_map = np.zeros_like(self.obst_map)
            self.obst_map[7, 10] = 1.0
            self.obst_map[5, 11] = 1.0
            self.obst_pos_arr = np.array([[1.25, 0.25, 2.5], [1.75, 1.25, 2.5]])
            self.obst_num = 2
        else:
            obst_generation_params = {
                'transpose_obst_area_flag': transpose_obst_area_flag,
                'obst_spawn_area': self.obst_spawn_area,
                'obst_grid_size': self.obst_grid_size,
                'obst_density': self.obst_density,
                'obst_size_range': self.obst_size_range,
                'min_gap_threshold': self.obst_min_gap_threshold,
                'obst_spawn_center': self.obst_spawn_center,
                'room_dims': self.room_dims
            }
            self.obst_map, self.obst_pos_arr, self.cell_centers, self.obst_num, self.obst_size_arr = obst_generation_given_density(
                params=obst_generation_params
            )

    def get_quads_sdf_obs(self, quads_pos):
        quads_sdf_obs = 100 * np.ones((len(quads_pos), 9))
        quads_sdf_obs = get_surround_sdfs(
            quad_poses=quads_pos[:, :2], obst_poses=self.obst_pos_arr[:, :2], quads_sdf_obs=quads_sdf_obs,
            obst_size_arr=self.obst_size_arr, resolution=self.resolution
        )

        return quads_sdf_obs

    def get_quads_tof_obs(self, quads_pos, quads_rots):
        noise_angles = self.scan_angle_arr + np.random.normal(
            loc=0, scale=self.angle_noise_std, size=self.scan_angle_arr.shape)
        quads_tof_obs = get_ToFs_depthmap(
            quad_poses=quads_pos, obst_poses=self.obst_pos_arr, obst_size_arr=self.obst_size_arr,
            scan_max_dist=self.range_max, quad_rotations=quads_rots, scan_angle_arr=noise_angles,
            fov_angle=self.fov_angle, num_rays=self.num_rays
        )
        quads_tof_obs = quads_tof_obs + np.random.uniform(
            low=-self.obst_noise * quads_tof_obs, high=self.obst_noise * quads_tof_obs, size=quads_tof_obs.shape
        )
        quads_tof_obs = np.clip(quads_tof_obs, a_min=0.0, a_max=self.range_max)

        return quads_tof_obs

    def reset(self, obs, quads_pos, quads_rots=None):
        if self.obst_obs_type == 'octomap':
            quads_obst_obs = self.get_quads_sdf_obs(quads_pos=quads_pos)
        elif self.obst_obs_type == 'ToFs':
            quads_obst_obs = self.get_quads_tof_obs(quads_pos=quads_pos, quads_rots=quads_rots)
            self.prev = np.copy(quads_obst_obs)
            self.tick = 0
        else:
            raise ValueError(f"Unknown obstacle observation type: {self.obst_obs_type}")

        if self.use_obst_octomap_critic:
            quads_obs_critic = self.get_quads_sdf_obs(quads_pos=quads_pos)
            obs = np.concatenate((obs, quads_obst_obs, quads_obs_critic), axis=1)
        else:
            obs = np.concatenate((obs, quads_obst_obs), axis=1)

        return obs

    def step(self, obs, quads_pos, quads_rots=None):
        if self.obst_obs_type == 'octomap':
            quads_obst_obs = self.get_quads_sdf_obs(quads_pos=quads_pos)
        elif self.obst_obs_type == 'ToFs':
            self.tick += 1
            if self.tick % self.sample_freq == 0:
                quads_obst_obs = self.get_quads_tof_obs(quads_pos=quads_pos, quads_rots=quads_rots)
                self.prev = np.copy(quads_obst_obs)
                self.tick = 0
            else:
                quads_obst_obs = np.copy(self.prev)
        else:
            raise ValueError(f"Unknown obstacle observation type: {self.obst_obs_type}")

        if self.use_obst_octomap_critic:
            quads_obs_critic = self.get_quads_sdf_obs(quads_pos=quads_pos)
            obs = np.concatenate((obs, quads_obst_obs, quads_obs_critic), axis=1)
        else:
            obs = np.concatenate((obs, quads_obst_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        quad_collisions = collision_detection(
            quad_poses=pos_quads[:, :2], obst_poses=self.obst_pos_arr[:, :2], obst_size_arr=self.obst_size_arr,
            quad_radius=self.quad_radius
        )

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
