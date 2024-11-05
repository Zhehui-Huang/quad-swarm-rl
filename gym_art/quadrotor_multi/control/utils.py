import numpy as np
from numba import njit

# jacobian of (acceleration magnitude, angular acceleration)
#       w.r.t (normalized motor thrusts) in range [0, 1]
def quadrotor_jacobian(dynamics):
    torque = dynamics.thrust_max * dynamics.prop_crossproducts.T
    torque[2, :] = dynamics.torque_max * dynamics.prop_ccw
    thrust = dynamics.thrust_max * np.ones((1, 4))
    dw = (1.0 / dynamics.inertia)[:, None] * torque
    dv = thrust / dynamics.mass
    J = np.vstack([dv, dw])
    J_cond = np.linalg.cond(J)
    # assert J_cond < 100.0
    if J_cond > 50:
        print("WARN: Jacobian conditioning is high: ", J_cond)
    return J

def _compute_maximum_distance_to_boundary(acc, G, h):
    """
        Computes the maximum distance of v to the constraint boundary
        defined by Gx <= h.

        If a constraint is satisfied by v, the distance of v to the
        constraint boundary is negative. Constraints take value zero at the
        boundary. Distance to violated constraints is positive.

        If the return value is positive, the return value is the distance
        to the most violated constraint.

        If it is non-positive, all constraints are satisfied,
        and the return value is the (negated) distance to the
        closest constraint.

        In any case, we want this value to be minimized.
    """
    G_norms = np.linalg.norm(G, axis=1)
    G_unit = G / G_norms[:, np.newaxis]
    h_norm = h / G_norms

    # Compute the distance of the acceleration to the constraint boundaries
    distances = np.dot(G_unit, acc) - h_norm
    return np.maximum(distances.max(), 0)


@njit
def get_min_real_dist(min_rel_dist, self_state_pos, description_state_pos, rel_pos_arr, rel_pos_norm_arr,
                      neighbor_des_num):
    for idx in range(neighbor_des_num):
        rel_pos = self_state_pos - description_state_pos[idx]
        rel_pos_norm = np.linalg.norm(rel_pos)

        rel_pos_arr[idx] = rel_pos
        rel_pos_norm_arr[idx] = rel_pos_norm
        min_rel_dist = min(rel_pos_norm, min_rel_dist)

    return min_rel_dist


@njit
def get_obst_min_real_dist(min_rel_dist, self_state_pos, description_state_pos, rel_pos_arr, rel_pos_norm_arr,
                           neighbor_des_num):
    for idx in range(neighbor_des_num):
        rel_pos = self_state_pos - description_state_pos[idx]
        rel_pos_norm = np.linalg.norm(rel_pos)

        rel_pos_arr[idx] = rel_pos
        rel_pos_norm_arr[idx] = rel_pos_norm
        min_rel_dist = min(rel_pos_norm, min_rel_dist)

    return min_rel_dist


@njit
def get_G_h(self_state_vel, neighbor_descriptions, rel_pos_arr, rel_pos_norm_arr, safety_distance,
            maximum_linf_acceleration, aggressiveness, G, h, start_id, max_lin_acc, neighbor_des_num):
    for idx in range(neighbor_des_num):
        rel_vel = self_state_vel - neighbor_descriptions[idx]
        safe_rel_dist = rel_pos_norm_arr[idx] - safety_distance
        rel_pos_dot_rel_vel = np.dot(rel_pos_arr[idx], rel_vel)

        # chunk 1
        norm_dp_dv = rel_pos_dot_rel_vel / rel_pos_norm_arr[idx]
        hij = np.sqrt(2.0 * max_lin_acc * safe_rel_dist) + norm_dp_dv
        chunk_1 = aggressiveness * (hij ** 3) * rel_pos_norm_arr[idx]

        # chunk 2
        chunk_2 = -(rel_pos_dot_rel_vel ** 2) / (rel_pos_norm_arr[idx] ** 2)

        # chunk 3
        chunk_3 = np.sqrt(max_lin_acc) * rel_pos_dot_rel_vel / np.sqrt(2.0 * safe_rel_dist)

        # chunk 4
        chunk_4 = np.dot(rel_vel, rel_vel)

        bij = chunk_1 + chunk_2 + chunk_3 + chunk_4
        bij_bar = (maximum_linf_acceleration / max_lin_acc) * bij

        G[start_id + idx] = -1.0 * rel_pos_arr[idx]
        h[start_id + idx] = bij_bar

    return G, h


@njit
def get_obst_G_h(self_state_vel, neighbor_descriptions, rel_pos_arr, rel_pos_norm_arr, safety_distance,
                 maximum_linf_acceleration, aggressiveness, G, h, start_id, max_lin_acc, neighbor_des_num):
    for idx in range(neighbor_des_num):
        # rel pos and rel vel is 2D
        rel_vel = self_state_vel - neighbor_descriptions[idx]
        safe_rel_dist = rel_pos_norm_arr[idx] - safety_distance
        rel_pos_dot_rel_vel = np.dot(rel_pos_arr[idx], rel_vel)

        # chunk 1
        norm_dp_dv = rel_pos_dot_rel_vel / rel_pos_norm_arr[idx]
        hij = np.sqrt(2.0 * max_lin_acc * safe_rel_dist) + norm_dp_dv
        chunk_1 = aggressiveness * (hij ** 3) * rel_pos_norm_arr[idx]

        # chunk 2
        chunk_2 = -(rel_pos_dot_rel_vel ** 2) / (rel_pos_norm_arr[idx] ** 2)

        # chunk 3
        chunk_3 = np.sqrt(max_lin_acc) * rel_pos_dot_rel_vel / np.sqrt(2.0 * safe_rel_dist)

        # chunk 4
        chunk_4 = np.dot(rel_vel, rel_vel)

        bij = chunk_1 + chunk_2 + chunk_3 + chunk_4
        bij_bar = (maximum_linf_acceleration / max_lin_acc) * bij

        tmp = -1.0 * rel_pos_arr[idx]
        G[start_id + idx] = np.array([tmp[0], tmp[1], 0.0])
        h[start_id + idx] = bij_bar

    return G, h


@njit
def get_G_h_room(self_state_vel, neighbor_descriptions, rel_pos_arr, rel_pos_norm_arr, safety_distance,
                 maximum_linf_acceleration, aggressiveness, G, h, start_id):
    max_lin_acc = maximum_linf_acceleration
    for idx in range(len(neighbor_descriptions)):
        rel_vel = self_state_vel[int(idx // 2)]
        rel_pos_dot_rel_vel = rel_pos_arr[idx] * rel_vel
        safe_rel_dist = rel_pos_norm_arr[idx] - safety_distance

        # chunk 1
        norm_dp_dv = rel_pos_dot_rel_vel / rel_pos_norm_arr[idx]
        hij = np.sqrt(2.0 * max_lin_acc * safe_rel_dist) + norm_dp_dv
        chunk_1 = aggressiveness * (hij ** 3) * rel_pos_norm_arr[idx]

        # chunk 2
        chunk_2 = -(rel_pos_dot_rel_vel ** 2) / (rel_pos_norm_arr[idx] ** 2)

        # chunk 3
        chunk_3 = np.sqrt(max_lin_acc) * rel_pos_dot_rel_vel / np.sqrt(2.0 * safe_rel_dist)

        # chunk 4
        chunk_4 = rel_vel ** 2

        bij = chunk_1 + chunk_2 + chunk_3 + chunk_4
        bij_bar = (maximum_linf_acceleration / max_lin_acc) * bij

        coefficients = np.zeros(3)
        coefficients[int(idx // 2)] = -1.0 * rel_pos_arr[idx]
        G[start_id + idx] = coefficients
        h[start_id + idx] = bij_bar

    return G, h

def vee(R):
    return np.array([R[2, 1], R[0, 2], R[1, 0]])