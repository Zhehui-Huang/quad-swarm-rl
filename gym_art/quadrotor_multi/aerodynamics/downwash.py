import numpy as np


def perform_downwash(drones_dyn, dt):
    # based on some data from Neural-Swarm: https://arxiv.org/pdf/2003.02992.pdf, Fig. 3
    # quadrotor weights: 34 grams
    # 0.5 m, force = 4 grams ; 0.4 m, force = 6 grams
    # 0.3 m, force = 8 grams ; 0.2 m, force = 10 grams
    # force function: f(x) = -20x + 14
    # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17, x in [0, 0.7]
    # Use cylinder to simulate the downwash area
    # The downwash area is a cylinder with radius of 2 arm ~ 10 cm and height of 1.0 m
    xy_downwash = 0.1
    z_downwash = 5.0
    # get pos
    dyns_pos = np.array([d.pos for d in drones_dyn])
    # get z_axis
    dyns_z_axis = np.array([d.rot[:, -1] for d in drones_dyn])
    # drone num
    dyns_num = len(drones_dyn)
    applied_downwash_list = np.zeros(dyns_num)
    # check if neighbors drones are within the downwash areas, if yes, apply downwash
    for i in range(dyns_num):
        z_axis = dyns_z_axis[i]
        neighbor_pos = dyns_pos - dyns_pos[i]
        neighbor_pos_dist = np.linalg.norm(neighbor_pos, axis=1)
        # acceleration func: a(x) = f(x) / 34 = -10 / 17 * x + 7 / 17
        # x in [0, 0.7], a(x) in [0.0, 0.41]
        # acc = (1 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.03, high=0.03)
        acc = (6 / 17) * (-10 * neighbor_pos_dist + 7) + np.random.uniform(low=-0.2, high=0.2)
        acc = np.maximum(1e-6, acc)

        # omega downwash given neighbor_pos_dist
        # 0.3 * (x - 1)^2 + random(-0.01, 0.01)
        omega_downwash = 0.3 * (neighbor_pos_dist - 1) ** 2 + np.random.uniform(low=-0.1, high=0.1)
        omega_downwash = np.maximum(1e-6, omega_downwash)

        rel_dists_z = np.dot(neighbor_pos, z_axis)
        rel_dists_xy = np.sqrt(neighbor_pos_dist ** 2 - rel_dists_z ** 2)

        for j in range(dyns_num):
            if i == j:
                continue

            if -z_downwash < rel_dists_z[j] < 0 and rel_dists_xy[j] < xy_downwash:
                down_z_axis_norm, dir_omega_norm = get_vel_omega_norm(z_axis=z_axis)
                drones_dyn[j].vel += acc[j] * down_z_axis_norm * dt
                drones_dyn[j].omega += omega_downwash[j] * dir_omega_norm * dt
                applied_downwash_list[j] = 1.0

    return applied_downwash_list


def get_vel_omega_norm(z_axis):
    # vel_norm
    noise_z_axis = z_axis + np.random.uniform(low=-0.1, high=0.1, size=3)
    noise_z_axis_mag = np.linalg.norm(noise_z_axis)
    noise_z_axis_norm = noise_z_axis / (noise_z_axis_mag + 1e-6 if noise_z_axis_mag == 0.0 else noise_z_axis_mag)
    down_z_axis_norm = -1.0 * noise_z_axis_norm

    # omega norm
    dir_omega = np.random.uniform(low=-1, high=1, size=3)
    dir_omega_mag = np.linalg.norm(dir_omega)
    dir_omega_norm = dir_omega / (dir_omega_mag + 1e-6 if dir_omega_mag == 0.0 else dir_omega_mag)

    return down_z_axis_norm, dir_omega_norm
