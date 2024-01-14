import numpy as np
import osqp
import scipy.sparse as spa
from gymnasium import spaces

from gym_art.quadrotor_multi.quad_utils import *
from gym_art.quadrotor_multi.quadrotor_control_utils import _compute_maximum_distance_to_boundary, get_min_real_dist, \
    get_G_h, get_G_h_room, get_obst_min_real_dist, get_obst_G_h

GRAV = 9.81


class NominalSBC:
    def __init__(self, maximum_linf_acceleration, radius, room_box, num_agents, num_obstacles):
        self.maximum_linf_acceleration = maximum_linf_acceleration
        self.sbc_neighbor_aggressive = 0.2
        self.sbc_obst_aggressive = 0.2
        self.sbc_room_aggressive = 0.2
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

    def plan(self, self_state, neighbor_descriptions, obstacle_descriptions, desired_acceleration,
             sbc_neighbor_aggressive, sbc_obst_aggressive, sbc_room_aggressive):
        self.sbc_neighbor_aggressive = sbc_neighbor_aggressive
        self.sbc_obst_aggressive = sbc_obst_aggressive
        self.sbc_room_aggressive = sbc_room_aggressive
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

            min_rel_dist = get_min_real_dist(min_rel_dist=min_rel_dist, self_state_pos=self_state['position'],
                                             description_state_pos=description_state_pos,
                                             rel_pos_arr=rel_pos_arr, rel_pos_norm_arr=rel_pos_norm_arr,
                                             neighbor_des_num=neighbor_des_num)
            safety_distance = self.radius + neighbor_descriptions[0]['radius']
            if min_rel_dist < safety_distance:
                return None, None
        else:
            safety_distance = None

        # Obstacle min_rel_dist check
        obst_des_num = len(obstacle_descriptions)
        obst_rel_pos_arr = np.zeros((obst_des_num, 2))
        obst_rel_pos_norm_arr = np.zeros(obst_des_num)
        if obst_des_num > 0:
            min_rel_dist = 100.0
            description_state_pos = -1 * np.ones((self.num_obstacles, 2))
            for idx in range(obst_des_num):
                description_state_pos[idx] = obstacle_descriptions[idx]['state']['position']

            min_rel_dist = get_obst_min_real_dist(
                min_rel_dist=min_rel_dist, self_state_pos=self_state['position'][:2],
                description_state_pos=description_state_pos, rel_pos_arr=obst_rel_pos_arr,
                rel_pos_norm_arr=obst_rel_pos_norm_arr, neighbor_des_num=obst_des_num)

            # # Obst min_rel_dist
            obst_safety_distance = self.radius + obstacle_descriptions[0]['radius']
            if min_rel_dist < obst_safety_distance:
                return None, None
        else:
            obst_safety_distance = None

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

            G, h = get_G_h(self_state_vel=self_state['velocity'],
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
                           neighbor_des_num=neighbor_des_num)
        # Obstacle G, h
        if obst_des_num > 0:
            obst_max_lin_acc = (self.maximum_linf_acceleration +
                                obstacle_descriptions[0]['maximum_linf_acceleration_lower_bound'])
            obst_description_state_vel = -1 * np.ones((self.num_obstacles, 2))
            for idx in range(obst_des_num):
                obst_description_state_vel[idx] = obstacle_descriptions[idx]['state']['velocity']

            G, h = get_obst_G_h(self_state_vel=self_state['velocity'][:2],
                                neighbor_descriptions=obst_description_state_vel,
                                rel_pos_arr=obst_rel_pos_arr,
                                rel_pos_norm_arr=obst_rel_pos_norm_arr,
                                safety_distance=obst_safety_distance,
                                maximum_linf_acceleration=self.maximum_linf_acceleration,
                                aggressiveness=self.sbc_obst_aggressive,
                                G=G,
                                h=h,
                                start_id=neighbor_des_num,
                                max_lin_acc=obst_max_lin_acc,
                                neighbor_des_num=obst_des_num)

        # for idx, rel_pos_axis in enumerate(room_rel_pos.flatten()):
        room_rel_pos_flat = room_rel_pos.flatten('F')
        G, h = get_G_h_room(self_state_vel=self_state['velocity'],
                            neighbor_descriptions=room_rel_pos_flat,
                            rel_pos_arr=room_rel_pos_flat,
                            rel_pos_norm_arr=abs(room_rel_pos_flat),
                            safety_distance=self.radius,
                            maximum_linf_acceleration=self.maximum_linf_acceleration,
                            aggressiveness=self.sbc_room_aggressive,
                            G=G,
                            h=h,
                            start_id=neighbor_des_num + obst_des_num)

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


class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        self.action = None
        self.step_func = self.step
        self.high = np.ones(4)
        self.low = -np.ones(4)
        self.bias = 1.0
        self.scale = 0.5

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0.0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(4)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()

    # @profile
    def step_tf(self, dynamics, action, goal, dt, observation=None):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()


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


class MellingerController(object):
    # @profile
    def __init__(self, dynamics, sbc_radius, room_box, num_agents, num_obstacles, sbc_max_acc, enable_sbc):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        self.action = None

        # self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0

        self.rot_des = np.eye(3)

        self.enable_sbc = enable_sbc
        # maximum_linf_acceleration, max_acc in any dimension
        if enable_sbc:
            self.sbc = NominalSBC(maximum_linf_acceleration=sbc_max_acc, radius=sbc_radius, room_box=room_box,
                                  num_agents=num_agents, num_obstacles=num_obstacles)
            self.sbc_last_safe_acc = np.zeros(3)

        self.step_func = self.step

        self.no_sol_flag = False

    def step(self, dynamics, acc_des, dt, observation=None):
        # Preset
        acc_rl = np.array(acc_des)
        acc_for_control = np.array(acc_des)
        sbc_distance_to_boundary = None
        if self.enable_sbc:
            no_sol_count = observation['no_sol']
            no_continuous_sol_count = observation['no_continuous_sol']
            modify_num = observation['modify_num']
            change_amount = observation['change_amount']
        else:
            no_sol_count = 0
            no_continuous_sol_count = 0
            modify_num = 0
            change_amount = []

        if self.enable_sbc and observation is not None:
            new_acc, sbc_distance_to_boundary = self.sbc.plan(
                self_state=observation["self_state"],
                neighbor_descriptions=observation["neighbor_descriptions"],
                obstacle_descriptions=observation["obstacle_descriptions"],
                desired_acceleration=acc_rl,
                sbc_neighbor_aggressive=observation["sbc_neighbor_aggressive"],
                sbc_obst_aggressive=observation["sbc_obst_aggressive"],
                sbc_room_aggressive=observation["sbc_room_aggressive"]
            )
            pre_no_sol_flag = self.no_sol_flag
            if new_acc is not None:
                self.sbc_last_safe_acc = np.array(new_acc)
                acc_for_control = np.array(new_acc)
                self.no_sol_flag = False
            else:
                acc_for_control = np.array(acc_rl)
                no_sol_count += 1
                self.no_sol_flag = True

            if pre_no_sol_flag is False and self.no_sol_flag is True:
                no_continuous_sol_count += 1

        else:
            acc_for_control = np.array(acc_des)

        if not np.allclose(a=acc_for_control, b=acc_rl, rtol=1e-3, atol=1e-3):
            modify_num += 1
            change_amount.append(np.linalg.norm(acc_rl - acc_for_control))

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

        return (self.action, acc_for_control_without_grav, sbc_distance_to_boundary, no_sol_count,
                no_continuous_sol_count, modify_num, change_amount, self.no_sol_flag)
