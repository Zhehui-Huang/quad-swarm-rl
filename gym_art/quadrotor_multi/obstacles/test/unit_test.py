import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import (
    collision_detection,
    get_cell_centers,
    get_lidar_rays_with_bounds,
    get_surround_sdfs,
    sample_spaced_obstacle_indices,
)


def test_get_surround_sdfs():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    quads_sdf_obs = 100 * np.ones((len(quad_poses), 9))

    # get_surround_sdfs
    dist = []
    for i, x in enumerate([-0.1, 0, 0.1]):
        for j, y in enumerate([-0.1, 0, 0.1]):
            tmp = np.linalg.norm([x - obst_poses[0][0], y - obst_poses[0][1]]) - 0.3
            dist.append(tmp)

    test_res = get_surround_sdfs(quad_poses, obst_poses, quads_sdf_obs, obst_radius=0.3, resolution=0.1)
    true_res = np.array(dist)
    assert test_res.all() == true_res.all()
    return


def test_collision_detection():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.array([[0.2, 0.]])
    # collision_detection
    quad_collisions = collision_detection(quad_poses, obst_poses, obst_radius=0.3, quad_radius=0.05)
    test_res = np.where(quad_collisions > -1)[0]
    true_res = np.array([0])
    assert test_res.all() == true_res.all()
    return


def test_get_lidar_rays_with_bounds():
    quad_poses = np.array([[0., 0.]])
    obst_poses = np.zeros((0, 2))
    lidar_obs = 100 * np.ones((len(quad_poses), 9))
    identity_rotations = np.array([[[1., 0.], [0., 1.]]])

    test_res = get_lidar_rays_with_bounds(
        quad_poses=quad_poses,
        obst_poses=obst_poses,
        lidar_obs=lidar_obs,
        obst_radius=0.2,
        room_dims=np.array([12., 12.]),
        ray_rotations=identity_rotations,
    )
    ray_angles = 2.0 * np.pi * np.arange(9) / 9.0
    true_res = np.array([[
        min(
            6.0 / abs(np.cos(angle)) if abs(np.cos(angle)) > 1e-8 else 100.0,
            6.0 / abs(np.sin(angle)) if abs(np.sin(angle)) > 1e-8 else 100.0,
        )
        for angle in ray_angles
    ]])
    assert np.allclose(test_res, true_res, atol=1e-5)

    near_wall_obs = 100 * np.ones((1, 9))
    near_wall_res = get_lidar_rays_with_bounds(
        quad_poses=np.array([[5.5, 0.]]),
        obst_poses=obst_poses,
        lidar_obs=near_wall_obs,
        obst_radius=0.2,
        room_dims=np.array([12., 12.]),
        ray_rotations=identity_rotations,
    )
    assert np.isclose(near_wall_res[0, 0], 0.5, atol=1e-5)
    assert near_wall_res[0, 8] > 0.5

    yaw_90_obs = 100 * np.ones((1, 9))
    yaw_90_rotations = np.array([[[0., -1.], [1., 0.]]])
    yaw_90_res = get_lidar_rays_with_bounds(
        quad_poses=np.array([[5.5, 0.]]),
        obst_poses=obst_poses,
        lidar_obs=yaw_90_obs,
        obst_radius=0.2,
        room_dims=np.array([12., 12.]),
        ray_rotations=yaw_90_rotations,
    )
    assert np.isclose(yaw_90_res[0, 0], 6.0, atol=1e-5)
    assert np.min(yaw_90_res) > 0.5
    assert np.min(yaw_90_res) < 0.6

    obstacle_angle = np.deg2rad(15.0)
    offset_obstacle = np.array([[2.0 * np.cos(obstacle_angle), 2.0 * np.sin(obstacle_angle)]])
    single_ray_res = get_lidar_rays_with_bounds(
        quad_poses=quad_poses,
        obst_poses=offset_obstacle,
        lidar_obs=100 * np.ones((1, 9)),
        obst_radius=0.1,
        room_dims=np.array([12., 12.]),
        ray_rotations=identity_rotations,
    )
    sector_res = get_lidar_rays_with_bounds(
        quad_poses=quad_poses,
        obst_poses=offset_obstacle,
        lidar_obs=100 * np.ones((1, 9)),
        obst_radius=0.1,
        room_dims=np.array([12., 12.]),
        ray_rotations=identity_rotations,
        sector_angle=np.deg2rad(30.0),
        sector_samples=5,
    )
    assert single_ray_res[0, 0] > 5.0
    assert np.isclose(sector_res[0, 0], 1.9, atol=1e-5)

    ninth_sector_angle = 2.0 * np.pi * 8.0 / 9.0
    ninth_sector_obstacle = np.array([[
        2.0 * np.cos(ninth_sector_angle),
        2.0 * np.sin(ninth_sector_angle),
    ]])
    ninth_sector_res = get_lidar_rays_with_bounds(
        quad_poses=quad_poses,
        obst_poses=ninth_sector_obstacle,
        lidar_obs=100 * np.ones((1, 9)),
        obst_radius=0.1,
        room_dims=np.array([12., 12.]),
        ray_rotations=identity_rotations,
    )
    assert np.isclose(ninth_sector_res[0, 8], 1.9, atol=1e-5)
    return


def test_get_cell_centers():
    obst_area_length = 8.0
    obst_area_width = 8.0
    grid_size = 1.0
    test_res = get_cell_centers(obst_area_length=obst_area_length, obst_area_width=obst_area_width, grid_size=grid_size)

    true_res = np.array([
        (i + (grid_size / 2) - obst_area_length // 2, j + (grid_size / 2) - obst_area_width // 2)
        for i in np.arange(0, obst_area_length, grid_size)
        for j in np.arange(obst_area_width - grid_size, -grid_size, -grid_size)])

    assert test_res.all() == true_res.all()
    return


def test_sample_spaced_obstacle_indices():
    candidate_positions = get_cell_centers(10.0, 10.0, grid_size=1.0)
    selected = sample_spaced_obstacle_indices(
        candidate_positions=candidate_positions,
        num_obstacles=16,
        min_center_distance=1.2,
    )
    selected_positions = candidate_positions[selected]

    assert len(selected) == 16
    for i in range(len(selected_positions)):
        for j in range(i + 1, len(selected_positions)):
            assert np.linalg.norm(selected_positions[i] - selected_positions[j]) >= 1.2


def unit_test():
    test_get_surround_sdfs()
    test_collision_detection()
    test_get_lidar_rays_with_bounds()
    test_get_cell_centers()
    test_sample_spaced_obstacle_indices()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
