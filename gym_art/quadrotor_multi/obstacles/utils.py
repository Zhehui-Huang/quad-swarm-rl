import numpy as np
from numba import njit

@njit
def get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius, resolution=0.1):
    # Shape of quads_sdf_obs: (quad_num, 9)

    sdf_map = np.array([-1., -1., -1., 0., 0., 0., 1., 1., 1.])
    sdf_map *= resolution

    for i, q_pos in enumerate(quad_poses):
        q_pos_x, q_pos_y = q_pos[0], q_pos[1]

        for g_i, g_x in enumerate([q_pos_x - resolution, q_pos_x, q_pos_x + resolution]):
            for g_j, g_y in enumerate([q_pos_y - resolution, q_pos_y, q_pos_y + resolution]):
                grid_pos = np.array([g_x, g_y])

                min_dist = 100.0
                for o_pos in obst_poses:
                    dist = np.linalg.norm(grid_pos - o_pos)
                    if dist < min_dist:
                        min_dist = dist

                g_id = g_i * 3 + g_j
                quads_sdf_obs[i, g_id] = min_dist - obst_radius

    return quads_sdf_obs

@njit
def is_surface_in_cylinder_view(vector, q_pos, o_pos, o_radius, fov_angle):
    # Calculate the direction vector from the origin to the cylinder center
    direction_vector = o_pos - q_pos
    if np.linalg.norm(direction_vector) <= o_radius:
        return 0, None

    # Calculate the unit vector in the direction of the given view vector
    view_vector = vector / np.linalg.norm(vector)

    # Calculate the angle between the direction vector and the view vector
    angle = np.arccos(np.dot(direction_vector, view_vector) / (np.linalg.norm(direction_vector) * np.linalg.norm(view_vector)))

    if np.dot(direction_vector, view_vector) > 0:
        if angle <= fov_angle/2:
            return np.linalg.norm(direction_vector) - o_radius, 2*o_radius

        # Calculate the angle between the direction vector and the normal to the cylinder surface
        angle_to_surface = np.arcsin(o_radius / np.linalg.norm(direction_vector))

        edge_vector_1 = np.dot(np.array([[np.cos(angle_to_surface), -np.sin(angle_to_surface)],[np.sin(angle_to_surface), np.cos(angle_to_surface)]]), direction_vector)
        edge_vector_2 = np.dot(np.array([[np.cos(angle_to_surface), np.sin(angle_to_surface)],[-np.sin(angle_to_surface), np.cos(angle_to_surface)]]), direction_vector)

        edge_angle_1 = np.arccos(np.dot(edge_vector_1, view_vector) / (np.linalg.norm(edge_vector_1) * np.linalg.norm(view_vector)))
        edge_angle_2 = np.arccos(np.dot(edge_vector_2, view_vector) / (np.linalg.norm(edge_vector_2) * np.linalg.norm(view_vector)))

        # Case where edge is in FOV
        if edge_angle_1 <= fov_angle / 2 or edge_angle_2 <= fov_angle / 2:
            # Create a triangle with direction vector
            closest_dist = np.linalg.norm(direction_vector) * np.sin(angle - (fov_angle / 2))
            full_dist = np.linalg.norm(direction_vector) * np.cos(angle - (fov_angle / 2))
            if closest_dist <= o_radius:
                len_in_obst = (o_radius ** 2 - closest_dist ** 2) ** 0.5
                return full_dist - len_in_obst, 2 * len_in_obst
            else:
                return (None, None)
        # Case where full FOV is between center and edge
        if (np.dot(edge_vector_1, view_vector) > 0 and np.abs(angle_to_surface-(angle + edge_angle_1))<1e-5) or (
                np.dot(edge_vector_1, view_vector) > 0 and np.abs(angle_to_surface-(angle + edge_angle_2))<1e-5):
            # Create a triangle with direction vector
            closest_dist = np.linalg.norm(direction_vector) * np.sin(angle - (fov_angle / 2))
            full_dist = np.linalg.norm(direction_vector) * np.cos(angle - (fov_angle / 2))
            len_in_obst = (o_radius ** 2 - closest_dist ** 2) ** 0.5
            return full_dist - len_in_obst, 2 * len_in_obst
    return (None, None)

@njit
def get_ToFs_depthmap(quad_poses, obst_poses, obst_radius, scan_max_dist,
                              quad_rotations, scan_angle_arr, num_rays, fov_angle, obst_noise):
        """
            quad_poses:     quadrotor positions, only with xy pos
            obst_poses:     obstacle positions, only with xy pos
            quad_vels:      quadrotor velocities, only with xy vel
            obst_radius:    obstacle raidus
        """
        sensor_offset = 0.01625
        modifications = []
        for i in range(-num_rays+1, num_rays, 2):
            modifications.append(i*(fov_angle/ (num_rays*2)))
        modifications = np.array(modifications)
        quads_obs = scan_max_dist * np.ones((len(quad_poses), len(scan_angle_arr) * num_rays))

        for q_id in range(len(quad_poses)):
            q_pos_xy = quad_poses[q_id][:2]
            q_yaw = np.arctan2(quad_rotations[q_id][1, 0], quad_rotations[q_id][0, 0])
            base_rad = q_yaw
            walls = np.array([[5, q_pos_xy[1]], [-5, q_pos_xy[1]], [q_pos_xy[0], 5], [q_pos_xy[1], -5]])
            for ray_id, rad_shift in enumerate(scan_angle_arr):
                for sec_id, sec in enumerate(modifications):
                    cur_rad = base_rad + rad_shift + sec
                    cur_dir = np.array([np.cos(cur_rad), np.sin(cur_rad)])

                    # Check distances with wall
                    # for w_id in range(len(walls)):
                    #     wall_dir = walls[w_id] - q_pos_xy
                    #     if np.dot(wall_dir, cur_dir) > 0:
                    #         angle = np.arccos(
                    #             np.dot(wall_dir, cur_dir) / (np.linalg.norm(wall_dir) * np.linalg.norm(cur_dir)))
                    #
                    #         # Check if shortest line to wall is in FOV, else project to edge of FOV
                    #         if angle <= fov_angle / (num_rays*2):
                    #             quads_obs[q_id][ray_id*num_rays+sec_id] = min(quads_obs[q_id][ray_id*num_rays+sec_id], (np.linalg.norm(wall_dir)) - sensor_offset)
                    #         else:
                    #             quads_obs[q_id][ray_id*num_rays+sec_id] = min(quads_obs[q_id][ray_id*num_rays+sec_id], (np.linalg.norm(wall_dir) / np.cos(angle - (fov_angle / (num_rays*2)))) - sensor_offset)

                    # Check distances with obstacles
                    for o_id in range(len(obst_poses)):
                        o_pos_xy = obst_poses[o_id][:2]

                        # Returns distance and length of the path inside the circle along the shortest distance vector
                        distance, circle_len = is_surface_in_cylinder_view(cur_dir, q_pos_xy, o_pos_xy, obst_radius,
                                                                           fov_angle / num_rays)
                        if distance is not None:
                            quads_obs[q_id][ray_id*num_rays+sec_id] = min(quads_obs[q_id][ray_id*num_rays+sec_id], distance-sensor_offset)

        quads_obs = quads_obs + np.random.normal(loc=0, scale=obst_noise, size=quads_obs.shape)
        quads_obs = np.clip(quads_obs, a_min=0.0, a_max=scan_max_dist)
        return quads_obs

@njit
def collision_detection(quad_poses, obst_poses, obst_radius, quad_radius):
    quad_num = len(quad_poses)
    collide_threshold = quad_radius + obst_radius
    # Get distance matrix b/w quad and obst
    quad_collisions = -1 * np.ones(quad_num)
    for i, q_pos in enumerate(quad_poses):
        for j, o_pos in enumerate(obst_poses):
            dist = np.linalg.norm(q_pos - o_pos)
            if dist <= collide_threshold:
                quad_collisions[i] = j
                break

    return quad_collisions


@njit
def get_cell_centers(obst_area_length, obst_area_width, grid_size=1.):
    count = 0
    i_len = obst_area_length // grid_size
    j_len = obst_area_width // grid_size
    cell_centers = np.zeros((int(i_len * j_len), 2))
    for i in np.arange(0, i_len * grid_size, grid_size):
        for j in np.arange(j_len * grid_size - grid_size, -grid_size, -grid_size):
            cell_centers[count][0] = i + (grid_size / 2) - obst_area_length // 2
            cell_centers[count][1] = j + (grid_size / 2) - obst_area_width // 2
            count += 1

    return cell_centers


if __name__ == "__main__":
    from gym_art.quadrotor_multi.obstacles.test.unit_test import unit_test
    from gym_art.quadrotor_multi.obstacles.test.speed_test import speed_test

    # Unit Test
    unit_test()
    speed_test()
