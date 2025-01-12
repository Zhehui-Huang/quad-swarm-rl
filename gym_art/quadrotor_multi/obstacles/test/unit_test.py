import numpy as np

from gym_art.quadrotor_multi.obstacles.utils import get_surround_sdfs, collision_detection, get_cell_centers, \
    get_ToFs_depthmap


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
    quad_collisions = collision_detection(quad_poses, obst_poses, obst_radius=0.3)
    test_res = np.where(quad_collisions > -1)[0]
    true_res = np.array([0])
    assert test_res.all() == true_res.all()
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


def test_get_ToFs_depthmap():
    quad_poses = np.array([[0., 0., 0.]])
    obst_poses = np.array([[1., 0., 5.]])
    obst_size_arr = np.array([1.0])
    scan_max_dist = 2.0
    quad_rotations = np.array([np.identity(3)])
    scan_angle_arr = np.array([0., np.pi/2, np.pi, -np.pi/2])
    num_rays = 4
    fov_angle = 45 * np.pi / 180

    quads_obs = get_ToFs_depthmap(
        quad_poses, obst_poses, obst_size_arr, scan_max_dist, quad_rotations, scan_angle_arr, num_rays, fov_angle
    )
    print(quads_obs)



def unit_test():
    test_get_surround_sdfs()
    test_collision_detection()
    test_get_cell_centers()
    test_get_ToFs_depthmap()
    print('Pass unit test!')
    return


if __name__ == "__main__":
    unit_test()
