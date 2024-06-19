import numpy as np
from numba import njit

from gym_art.quadrotor_multi.quad_utils import EPS
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


@njit
def compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2):
    # Ge the collision normal, i.e difference in position
    collision_norm = pos1 - pos2
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    v1new = np.dot(vel1, collision_norm)
    v2new = np.dot(vel2, collision_norm)

    return v1new, v2new, collision_norm


@njit
def perform_collision_between_drones(pos1, vel1, omega1, pos2, vel2, omega2):
    # Solve for the new velocities using the elastic collision equations.
    # vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    v1new, v2new, collision_norm = compute_col_norm_and_new_velocities(pos1, vel1, pos2, vel2)
    vel_change = (v2new - v1new) * collision_norm
    dyn1_vel_shift = vel_change
    dyn2_vel_shift = -vel_change

    # Make sure new vel direction would be opposite to the original vel direction
    for _ in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.8, size=3)
        vel1_noise = cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)
        vel2_noise = -cons_rand_val + np.random.normal(loc=0, scale=0.15, size=3)

        dyn1_vel_shift = vel_change + vel1_noise
        dyn2_vel_shift = -vel_change + vel2_noise

        dyn1_new_vel_dir = np.dot(vel1 + dyn1_vel_shift, collision_norm)
        dyn2_new_vel_dir = np.dot(vel2 + dyn2_vel_shift, collision_norm)

        if dyn1_new_vel_dir > 0 > dyn2_new_vel_dir:
            break

    # Get new vel
    max_vel_magn = max(np.linalg.norm(vel1), np.linalg.norm(vel2))
    vel1 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel1, vel_shift=dyn1_vel_shift)
    vel2 = compute_new_vel(max_vel_magn=max_vel_magn, vel=vel2, vel_shift=dyn2_vel_shift)

    # Get new omega
    new_omega = compute_new_omega()
    omega1 += new_omega
    omega2 -= new_omega

    return vel1, omega1, vel2, omega2


@njit
def calculate_collision_matrix(positions, collision_threshold):
    """
    drone_col_matrix: set collided quadrotors' id to 1
    curr_drone_collisions: [i, j]
    distance_matrix: [i, j, dist]
    """
    num_agents = len(positions)
    item_num = int(num_agents * (num_agents - 1) / 2)
    count = int(0)

    drone_col_matrix = -1000 * np.ones(num_agents)
    curr_drone_collisions = -1000. * np.ones((item_num, 2))
    distance_matrix = -1000. * np.ones((item_num, 3))

    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            distance_matrix[count] = [i, j,
                                      ((positions[i][0] - positions[j][0]) ** 2 +
                                       (positions[i][1] - positions[j][1]) ** 2 +
                                       (positions[i][2] - positions[j][2]) ** 2) ** 0.5]

            if distance_matrix[count][2] <= collision_threshold:
                drone_col_matrix[i] = 1
                drone_col_matrix[j] = 1
                curr_drone_collisions[count] = [i, j]

            count += 1

    return drone_col_matrix, curr_drone_collisions, distance_matrix


@njit
def calculate_drone_proximity_penalties(distance_matrix, collision_falloff_threshold, dt, max_penalty, num_agents):
    penalties = np.zeros(num_agents)
    penalty_ratio = -max_penalty / collision_falloff_threshold
    for i, j, dist in distance_matrix:
        penalty = penalty_ratio * dist + max_penalty
        penalties[int(i)] += penalty
        penalties[int(j)] += penalty

    return dt * penalties


def calculate_drone_obst_proximity_penalties(r_drone, r_obst, penalty_coeff, penalty_range,
                                             quads_pos, quads_vel, obst_pos, dt):
    penalties = np.zeros(len(quads_pos))
    for qid in range(len(quads_pos)):
        q_vel = quads_vel[qid]
        q_speed = np.linalg.norm(q_vel)
        penalty_item = 0.0
        for oid in range(len(obst_pos)):
            rel_pos = obst_pos[oid] - quads_pos[qid]
            dist = np.linalg.norm(rel_pos)
            # theta: angle between q_vel and obst_pos - q_pos
            cos_theta = np.dot(q_vel, rel_pos) / (q_speed * dist)
            theta = np.arccos(cos_theta)
            # alpha: (r_drone + r_obst) / dist
            sin_alpha = (r_drone + r_obst) / dist
            alpha = np.arcsin(sin_alpha)

            penalty_bool = float(theta <= alpha) * float(dist <= (penalty_range + r_drone + r_obst))
            tmp_penalty = penalty_bool * penalty_coeff * (q_speed * cos_theta)
            penalty_item = max(penalty_item, tmp_penalty)

        penalties[qid] = penalty_item

    return dt * penalties


def unit_test():
    penalties = calculate_drone_obst_proximity_penalties(
        r_drone=0.05, r_obst=0.15, penalty_coeff=2, penalty_range=1, quads_pos=np.array([[0., 0., 2.]]),
        quads_vel=np.array([[1., 0., 0.]]), obst_pos=np.array([[0.5, 0., 2.]]), dt=1)
    print(penalties)
    return


if __name__ == "__main__":
    unit_test()