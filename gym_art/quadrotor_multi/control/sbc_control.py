import osqp
import scipy.sparse as spa

from gym_art.quadrotor_multi.control.utils import _compute_maximum_distance_to_boundary, get_min_real_dist, \
    get_G_h, get_G_h_room, get_obst_min_real_dist, get_obst_G_h, quadrotor_jacobian
from gym_art.quadrotor_multi.quad_utils import *

GRAV = 9.81


class NominalSBC:
    def __init__(self, maximum_linf_acceleration, radius, room_box, num_agents, num_obstacles,
                 sbc_neighbor_aggressive, sbc_obst_aggressive, sbc_room_aggressive):
        self.maximum_linf_acceleration = maximum_linf_acceleration
        self.sbc_neighbor_aggressive = sbc_neighbor_aggressive
        self.sbc_obst_aggressive = sbc_obst_aggressive
        self.sbc_room_aggressive = sbc_room_aggressive
        self.radius = radius
        self.room_box = room_box
        P = 2.0 * np.eye(3)
        A = np.ndarray((0, 3), np.float64)

        self.P = spa.csc_matrix(P)
        self.A = spa.csc_matrix(A)
        self.b = np.array([])
        self.lb = np.array([-self.maximum_linf_acceleration] * 3)
        self.ub = np.array([self.maximum_linf_acceleration] * 3)
        self.A_osqp = spa.eye(3)
        self.num_agents = num_agents
        self.num_obstacles = num_obstacles

    def plan(self, self_state, neighbor_descriptions, obstacle_descriptions, desired_acceleration):
        # Add robot and robot / obstacle collision avoidance constraints
        min_rel_dist = 100.0
        # Neighbor min_rel_dist check
        neighbor_des_num = len(neighbor_descriptions)
        rel_pos_arr = np.zeros((neighbor_des_num, 3))
        rel_pos_norm_arr = np.zeros(neighbor_des_num)
        if neighbor_des_num > 0:
            description_state_pos = -1 * np.ones((self.num_agents, 3))
            for idx in range(neighbor_des_num):
                description_state_pos[idx] = neighbor_descriptions[idx]['state']['position']

            min_rel_dist = get_min_real_dist(
                min_rel_dist=min_rel_dist, self_state_pos=self_state['position'],
                description_state_pos=description_state_pos, rel_pos_arr=rel_pos_arr, rel_pos_norm_arr=rel_pos_norm_arr,
                neighbor_des_num=neighbor_des_num
            )
            safety_distance = self.radius + neighbor_descriptions[0]['radius']
            if min_rel_dist <= safety_distance:
                return None, None
        else:
            safety_distance = None

        obst_des_num = len(obstacle_descriptions)
        if obst_des_num > 0:
            obst_rel_pos_arr = np.zeros((obst_des_num, 2))
            obst_rel_pos_norm_arr = np.zeros(obst_des_num)

            description_state_pos = -1 * np.ones((self.num_obstacles, 2))
            obst_safety_distance_arr = np.zeros(obst_des_num)
            for idx in range(obst_des_num):
                description_state_pos[idx] = obstacle_descriptions[idx]['state']['position']
                obst_safety_distance_arr[idx] = self.radius + obstacle_descriptions[idx]['radius']

            obst_rel_pos_arr, obst_rel_pos_norm_arr = get_obst_min_real_dist(
                self_state_pos=self_state['position'][:2],
                description_state_pos=description_state_pos, rel_pos_arr=obst_rel_pos_arr,
                rel_pos_norm_arr=obst_rel_pos_norm_arr, neighbor_des_num=obst_des_num)

            for real_dist, safe_dist in zip(obst_rel_pos_norm_arr, obst_safety_distance_arr):
                if real_dist <= safe_dist:
                    return None, None
        else:
            obst_safety_distance_arr = None

        # Room box
        # Add room box constraints
        room_rel_pos = np.array([
            self_state['position'] - self.room_box[0],
            self_state['position'] - self.room_box[1]
        ])
        if np.min(abs(room_rel_pos)) < self.radius:
            return None, None

        # 6 is room info, 6 faces
        G = np.zeros(shape=(neighbor_des_num + obst_des_num + 6, 3), dtype=np.float64)
        h = np.zeros(shape=(neighbor_des_num + obst_des_num + 6), dtype=np.float64)
        # Neighbor G, h
        if neighbor_des_num > 0:
            max_lin_acc = (self.maximum_linf_acceleration +
                           neighbor_descriptions[0]['maximum_linf_acceleration_lower_bound'])
            description_state_vel = -1 * np.ones((self.num_agents, 3))
            for idx in range(neighbor_des_num):
                description_state_vel[idx] = neighbor_descriptions[idx]['state']['velocity']

            G, h = get_G_h(
                self_state_vel=self_state['velocity'],
                neighbor_descriptions=description_state_vel,
                rel_pos_arr=rel_pos_arr,
                rel_pos_norm_arr=rel_pos_norm_arr,
                safety_distance=safety_distance,
                maximum_linf_acceleration=self.maximum_linf_acceleration,
                aggressiveness=self.sbc_neighbor_aggressive,
                G=G,
                h=h,
                start_id=0,
                max_lin_acc=max_lin_acc,
                neighbor_des_num=neighbor_des_num
            )

        # Obstacle G, h
        if obst_des_num > 0:
            obst_max_lin_acc = (self.maximum_linf_acceleration +
                                obstacle_descriptions[0]['maximum_linf_acceleration_lower_bound'])
            obst_description_state_vel = -1 * np.ones((self.num_obstacles, 2))
            for idx in range(obst_des_num):
                obst_description_state_vel[idx] = obstacle_descriptions[idx]['state']['velocity']

            G, h = get_obst_G_h(
                self_state_vel=self_state['velocity'][:2],
                neighbor_descriptions=obst_description_state_vel,
                rel_pos_arr=obst_rel_pos_arr,
                rel_pos_norm_arr=obst_rel_pos_norm_arr,
                safety_distance_arr=obst_safety_distance_arr,
                maximum_linf_acceleration=self.maximum_linf_acceleration,
                aggressiveness=self.sbc_obst_aggressive,
                G=G,
                h=h,
                start_id=neighbor_des_num,
                max_lin_acc=obst_max_lin_acc,
                neighbor_des_num=obst_des_num
            )

        # for idx, rel_pos_axis in enumerate(room_rel_pos.flatten()):
        room_rel_pos_flat = room_rel_pos.flatten('F')
        G, h = get_G_h_room(
            self_state_vel=self_state['velocity'],
            neighbor_descriptions=room_rel_pos_flat,
            rel_pos_arr=room_rel_pos_flat,
            rel_pos_norm_arr=abs(room_rel_pos_flat),
            safety_distance=self.radius,
            maximum_linf_acceleration=self.maximum_linf_acceleration,
            aggressiveness=self.sbc_room_aggressive,
            G=G,
            h=h,
            start_id=neighbor_des_num + obst_des_num
        )

        maximum_distance = _compute_maximum_distance_to_boundary(desired_acceleration, G, h)

        G = spa.csc_matrix(G)
        q = -2.0 * desired_acceleration

        A_osqp = None
        l_osqp = None
        u_osqp = None
        if G is not None and h is not None:
            A_osqp = G
            l_osqp = np.full(h.shape, -np.infty)
            u_osqp = h
        E = spa.eye(q.shape[0])
        A_osqp = E if A_osqp is None else spa.vstack([A_osqp, E], format="csc")
        l_osqp = self.lb if l_osqp is None else np.hstack([l_osqp, self.lb])
        u_osqp = self.ub if u_osqp is None else np.hstack([u_osqp, self.ub])

        solver = osqp.OSQP()
        solver.setup(P=self.P, q=q, A=A_osqp, l=l_osqp, u=u_osqp, verbose=False)
        res = solver.solve()

        if res.info.status_val == osqp.constant("OSQP_SOLVED"):
            return res.x, maximum_distance
        else:
            return None, maximum_distance


class MellingerController(object):
    def __init__(self, dynamics, room_box, num_agents, num_obstacles, sbc_obst_agg):
        # Initial
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        self.action = None
        # self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0
        self.rot_des = np.eye(3)


        # SBC
        self.last_sbc_distance_to_boundary = 0.0
        self.step_func = self.step

        sbc_radius = 0.05
        sbc_max_acc = 2.0
        sbc_neighbor_aggressive = 0.2
        sbc_obst_aggressive = sbc_obst_agg
        sbc_room_aggressive = 0.2

        self.sbc = NominalSBC(
            maximum_linf_acceleration=sbc_max_acc, radius=sbc_radius, room_box=room_box,
            num_agents=num_agents, num_obstacles=num_obstacles, sbc_neighbor_aggressive=sbc_neighbor_aggressive,
            sbc_obst_aggressive=sbc_obst_aggressive, sbc_room_aggressive=sbc_room_aggressive
        )

    def reset(self):
        self.last_sbc_distance_to_boundary = 0.0

    def step(self, dynamics, dt, acc_des, observation):
        acc_rl = np.array(acc_des)

        new_acc, sbc_distance_to_boundary = self.sbc.plan(
            self_state=observation["self_state"],
            neighbor_descriptions=observation["neighbor_descriptions"],
            obstacle_descriptions=observation["obstacle_descriptions"],
            desired_acceleration=acc_rl
        )

        if new_acc is not None:
            acc_for_control = np.array(new_acc)
            no_sol_flag = False
        else:
            acc_for_control = np.array(acc_rl)
            no_sol_flag = True

        if sbc_distance_to_boundary is not None:
            self.last_sbc_distance_to_boundary = sbc_distance_to_boundary
        else:
            sbc_distance_to_boundary = self.last_sbc_distance_to_boundary

        # Question: Why do we need to do this???
        acc_for_control_without_grav = np.array(acc_for_control)
        acc_for_control_with_grav = acc_for_control + np.array([0, 0, GRAV])
        xc_des = self.rot_des[:, 0]

        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_for_control_with_grav)
        yb_des, _ = normalize(cross(zb_des, xc_des))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))
        R = dynamics.rot

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        e_R[2] *= 0.2  # slow down yaw dynamics
        e_w = dynamics.omega

        dw_des = -self.kp_a * e_R - self.kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        thrust_mag = np.dot(acc_for_control_with_grav, R[:, 2])

        des = np.append(thrust_mag, dw_des)

        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()

        return self.action, acc_for_control_without_grav, sbc_distance_to_boundary, no_sol_flag
