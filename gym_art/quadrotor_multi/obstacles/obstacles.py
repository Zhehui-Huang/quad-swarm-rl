import copy
import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import (
    get_lidar_rays_with_bounds,
    get_surround_sdfs,
    get_surround_sdfs_with_bounds,
    collision_detection,
)


class MultiObstacles:
    def __init__(self, obstacle_size=1.0, quad_radius=0.046, resolution=0.1,
                 room_dims=None, include_room_bounds=False,
                 lidar_sector_angle=0.0, lidar_sector_samples=1):
        if lidar_sector_angle < 0.0:
            raise ValueError("lidar_sector_angle must be non-negative")
        if lidar_sector_samples < 1:
            raise ValueError("lidar_sector_samples must be at least 1")
        self.size = obstacle_size
        self.obstacle_radius = obstacle_size / 2.0
        self.quad_radius = quad_radius
        self.pos_arr = []
        self.resolution = resolution
        self.room_dims = np.array(room_dims[:2], dtype=np.float64) if room_dims is not None else np.zeros(2)
        self.include_room_bounds = include_room_bounds
        self.lidar_sector_angle = np.deg2rad(lidar_sector_angle)
        self.lidar_sector_samples = lidar_sector_samples

    def _empty_sdf_obs(self, quads_pos):
        return 100 * np.ones((len(quads_pos), 9))

    def _identity_ray_rotations(self, quads_pos):
        rotations = np.zeros((len(quads_pos), 2, 2), dtype=np.float64)
        rotations[:, 0, 0] = 1.0
        rotations[:, 1, 1] = 1.0
        return rotations

    def sdf_obs(self, quads_pos, ray_rotations=None):
        quads_sdf_obs = self._empty_sdf_obs(quads_pos)
        if self.include_room_bounds:
            obst_poses = self.pos_arr[:, :2] if self.pos_arr.size > 0 else np.zeros((0, 2))
            if ray_rotations is None:
                ray_rotations = self._identity_ray_rotations(quads_pos)
            quads_sdf_obs = get_lidar_rays_with_bounds(
                quad_poses=quads_pos[:, :2], obst_poses=obst_poses, lidar_obs=quads_sdf_obs,
                obst_radius=self.obstacle_radius, room_dims=self.room_dims, ray_rotations=ray_rotations,
                sector_angle=self.lidar_sector_angle, sector_samples=self.lidar_sector_samples)
        elif self.pos_arr.size > 0:
            quads_sdf_obs = get_surround_sdfs(quad_poses=quads_pos[:, :2], obst_poses=self.pos_arr[:, :2],
                                              quads_sdf_obs=quads_sdf_obs, obst_radius=self.obstacle_radius,
                                              resolution=self.resolution)
        return quads_sdf_obs

    def reset(self, obs, quads_pos, pos_arr, ray_rotations=None):
        self.pos_arr = copy.deepcopy(np.array(pos_arr))
        quads_sdf_obs = self.sdf_obs(quads_pos, ray_rotations=ray_rotations)
        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def step(self, obs, quads_pos, ray_rotations=None):
        quads_sdf_obs = self.sdf_obs(quads_pos, ray_rotations=ray_rotations)
        obs = np.concatenate((obs, quads_sdf_obs), axis=1)

        return obs

    def collision_detection(self, pos_quads):
        if self.pos_arr.size == 0:
            return np.array([], dtype=np.int64), {}

        quad_collisions = collision_detection(quad_poses=pos_quads[:, :2], obst_poses=self.pos_arr[:, :2],
                                              obst_radius=self.obstacle_radius, quad_radius=self.quad_radius)

        collided_quads_id = np.where(quad_collisions > -1)[0]
        collided_obstacles_id = quad_collisions[collided_quads_id]
        quad_obst_pair = {}
        for i, key in enumerate(collided_quads_id):
            quad_obst_pair[key] = int(collided_obstacles_id[i])

        return collided_quads_id, quad_obst_pair
