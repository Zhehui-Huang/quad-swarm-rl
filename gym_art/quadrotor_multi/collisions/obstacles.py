import numpy as np
from numba import njit

from gym_art.quadrotor_multi.quad_utils import EPS
from gym_art.quadrotor_multi.collisions.utils import compute_new_vel, compute_new_omega


@njit
def compute_col_norm_and_new_vel_obst(pos, vel, obstacle_pos):
    collision_norm = pos - obstacle_pos
    # difference in z position is 0, given obstacle height is same as room height
    collision_norm[2] = 0.0
    coll_norm_mag = np.linalg.norm(collision_norm)
    collision_norm = collision_norm / (coll_norm_mag + EPS if coll_norm_mag == 0.0 else coll_norm_mag)

    # Get the components of the velocity vectors which are parallel to the collision.
    # The perpendicular component remains the same.
    vnew = np.dot(vel, collision_norm)

    return vnew, collision_norm


def perform_collision_with_obstacle(drone_dyn, obstacle_pos, obstacle_size):
    # Vel noise has two different random components,
    # One that preserves momentum in opposite directions
    # Second that does not preserve momentum
    vnew, collision_norm = compute_col_norm_and_new_vel_obst(drone_dyn.pos, drone_dyn.vel, obstacle_pos)
    vel_magn = np.linalg.norm(drone_dyn.vel)
    new_vel = vel_magn * collision_norm

    vel_noise = np.zeros(3)
    for i in range(3):
        cons_rand_val = np.random.normal(loc=0, scale=0.1, size=3)
        tmp_vel_noise = cons_rand_val + np.random.normal(loc=0, scale=0.05, size=3)
        if np.dot(new_vel + tmp_vel_noise, collision_norm) > 0:
            vel_noise = tmp_vel_noise
            break

    max_vel_magn = np.linalg.norm(drone_dyn.vel)
    # In case drone that is inside the obstacle
    if np.linalg.norm(drone_dyn.pos - obstacle_pos) < obstacle_size / 2:
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel,
                                        vel_shift=new_vel - drone_dyn.vel + vel_noise, low=1.0, high=1.0)
    else:
        drone_dyn.vel = compute_new_vel(max_vel_magn=max_vel_magn, vel=drone_dyn.vel,
                                        vel_shift=new_vel - drone_dyn.vel + vel_noise)

    # Random forces for omega
    new_omega = compute_new_omega(magn_scale=1.0)
    drone_dyn.omega += new_omega
