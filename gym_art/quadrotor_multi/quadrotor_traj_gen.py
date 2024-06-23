import numpy as np
import copy
from quad_utils import normalize, cross_vec, norm2
from quadrotor_planner import QuadPlanner, poly4d, traj_eval

class QuadTrajGen:
    """ 
        This class generates a trajectory given an initial point and FINAL goal point. 
    """
    def __init__(
        self,
        poly_degree: int
    ):

        self.planner = QuadPlanner(poly_degree=7)
        self.poly_degree = 7

    def plan_go_to_from(
        self,
        current_eval: traj_eval,
        desired_state: "np.ndarray[np.float_]",
        duration: float,
        current_time: float 
    ):
        """ 
        Note: Python adaptation from plan_go_to_from() of the crazyflie firmware.
        
        Arguments:
            initial_state: [x, y, z, vx, vy, vz, roll, pitch, yaw, omega_x, omega_y, omega_z]
            desired_state: [x, y, z, yaw]
        """

        curr_yaw = self.__normalize_radians(current_eval.yaw)
        desired_yaw = self.__normalize_radians(desired_state[3])
        goal_yaw = curr_yaw + self.__smallest_signed_angle(curr_yaw, desired_yaw)

        hover_pos = np.array([desired_state[0], desired_state[1], desired_state[2]])

        self.__piecewise_plan_7th_order_no_jerk(duration, current_eval.pos, curr_yaw, current_eval.vel, 
        current_eval.omega[2], current_eval.acc, hover_pos, goal_yaw, np.zeros(3), 0, np.zeros(3))

        # self.planner.planned_trajectory["pieces"][0].poly[0] = np.array([0.000000, -0.000000, 0.000000, -0.000000, 0.830443, -0.276140, -0.384219, 0.180493])
        # self.planner.planned_trajectory["pieces"][0].poly[1] = np.array([-0.000000, 0.000000, -0.000000, 0.000000, -1.356107, 0.688430, 0.587426, -0.329106])
        # self.planner.planned_trajectory["pieces"][0].poly[2] = np.array([ 0, 0, 0, 0, 0, 0, 0, 0])
        # self.planner.planned_trajectory["pieces"][0].poly[3] = np.array([ 0, 0, 0, 0, 0, 0, 0, 0])
        self.planner.planned_trajectory["t_begin"] = current_time

    def piecewise_eval(self, t):
        cursor = 0
        t = t - self.planner.planned_trajectory["t_begin"]
        while (cursor < self.planner.planned_trajectory["n_pieces"]):
            piece = self.planner.planned_trajectory["pieces"][cursor]

            if (t <= piece.duration * self.planner.planned_trajectory["timescale"]):
                poly4d_tmp = piece
                self.__poly4d_shift(poly4d_tmp, self.planner.planned_trajectory["shift"][0], self.planner.planned_trajectory["shift"][1], self.planner.planned_trajectory["shift"][2], 0)
                self.__poly4d_stretchtime(poly4d_tmp, self.planner.planned_trajectory["timescale"])
                
                return self.__poly4d_eval(poly4d_tmp, t)
            
            t -= (piece.duration * self.planner.planned_trajectory["timescale"])
            cursor += 1

        end_piece = self.planner.planned_trajectory["pieces"][self.planner.planned_trajectory["n_pieces"] - 1]
        ev = self.__poly4d_eval(end_piece, end_piece.duration)
        ev.pos = np.add(ev.pos, self.planner.planned_trajectory["shift"])
        ev.vel = np.zeros(3)
        ev.acc = np.zeros(3)
        ev.omega = np.zeros(3)
        
        return ev

    def __piecewise_plan_7th_order_no_jerk(self, duration, p0, y0, v0, dy0, a0, p1, y1, v1, dy1, a1):
        p = poly4d(self.poly_degree)
        p.duration = duration

        self.planner.planned_trajectory["timescale"] = 1.0
        self.planner.planned_trajectory["shift"] = np.zeros(3)
        self.planner.planned_trajectory["n_pieces"] = 1

        self.__poly7_nojerk(p.poly[0], duration, p0[0], v0[0], a0[0], p1[0], v1[0], a1[0])
        self.__poly7_nojerk(p.poly[1], duration, p0[1], v0[1], a0[1], p1[1], v1[1], a1[1])
        self.__poly7_nojerk(p.poly[2], duration, p0[2], v0[2], a0[2], p1[2], v1[2], a1[2])
        self.__poly7_nojerk(p.poly[3], duration, y0, dy0, 0, y1, dy1, 0)

        self.planner.planned_trajectory["pieces"][0] = p

    def __poly7_nojerk(self, poly, T, x0, dx0, ddx0, xf, dxf, ddxf):

        if (T <= 0.0):
            poly[0] = xf
            poly[1] = dxf
            poly[2] = ddxf/2
            for i in range(3, self.planner.poly_degree):
                poly[i] = 0
        else:
            T2 = T * T
            T3 = T2 * T
            T4 = T3 * T
            T5 = T4 * T
            T6 = T5 * T
            T7 = T6 * T
            poly[0] = x0
            poly[1] = dx0
            poly[2] = ddx0/2
            poly[3] = 0
            poly[4] = -(5*(14*x0 - 14*xf + 8*T*dx0 + 6*T*dxf + 2*T2*ddx0 - T2*ddxf))/(2*T4)
            poly[5] = (84*x0 - 84*xf + 45*T*dx0 + 39*T*dxf + 10*T2*ddx0 - 7*T2*ddxf)/T5
            poly[6] = -(140*x0 - 140*xf + 72*T*dx0 + 68*T*dxf + 15*T2*ddx0 - 13*T2*ddxf)/(2*T6)
            poly[7] = (2*(10*x0 - 10*xf + 5*T*dx0 + 5*T*dxf + T2*ddx0 - T2*ddxf))/T7
            for i in range(8, self.planner.poly_degree):
                poly[i] = 0
    
    def __poly4d_shift(self, piece: poly4d, x: float, y: float, z: float, yaw: float):
        piece.poly[0][0] += x
        piece.poly[1][0] += y
        piece.poly[2][0] += z
        piece.poly[3][0] += yaw

    def __poly4d_stretchtime(self, piece: poly4d, t: float):
        """ Scale the duration of a polynomial by factor t"""
        for i in range(4):
            # Stretch per state
            recip = 1 / t
            scale = recip

            state = piece.poly[i]
            for j in range(piece.degree):
                state[j] *= scale
                scale *= recip

        piece.duration *= t
    
    def __polyval_xyz(self, piece: poly4d, t: float):
        """
        Returns a xyz state evaluation of polynomial.
        """
        x = self.__polyval(piece.poly[0], t)
        y = self.__polyval(piece.poly[1], t)
        z = self.__polyval(piece.poly[2], t)

        return np.array([x, y, z])

    def __polyval_yaw(self, piece: poly4d, t: float) -> float:
        yaw = self.__polyval(piece.poly[3], t)

        return yaw

    def __polyval(self, polynomial: "np.ndarray[np.float_]", t: float):
        """
        Evaluates a polynomial based on Horners rule.
        """
        x = 0
        for i in range(self.poly_degree, -1, -1):

            x = x * t + polynomial[i]

        return x

    def __poly4d_derivative(self, piece: poly4d):
        for i in range(4):
            # Take derivative of each state
            state = piece.poly[i]
            for j in range(1, piece.degree):
                state[j-1] = j * state[j]

            state[piece.degree] = 0

    def __normalize_radians(self, angle: float) -> float:
        """ Normalize radians to the range: [-pi, pi]"""
        return np.arctan2(np.sin(angle),np.cos(angle))

    # https://stackoverflow.com/questions/1878907/how-can-i-find-the-smallest-difference-between-two-angles-around-a-point
    def __smallest_signed_angle(self, start, goal) -> float:
        start = (start - goal) % (2*3.14)
        goal = (goal - start) % (2*3.14)
        return -start if start < goal else goal

    def __vcross(self, a, b):
        return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])
            
    def __poly4d_eval(self, piece: poly4d, t: float) -> traj_eval:
        # Output of desired states
        out = traj_eval()
        out.pos = self.__polyval_xyz(piece, t)
        out.yaw = self.__polyval_yaw(piece, t)

        # piece = copy.deepcopy(piece)

        # 1st Derivative
        self.__poly4d_derivative(piece)
        out.vel = self.__polyval_xyz(piece, t)
        dyaw = self.__polyval_yaw(piece, t)

        # 2nd Derivative
        self.__poly4d_derivative(piece)
        out.acc = self.__polyval_xyz(piece, t)

        # 3rd Derivative
        self.__poly4d_derivative(piece)
        jerk = self.__polyval_xyz(piece, t)

        thrust = np.add(out.acc, [0, 0, 9.81])
        z_body, _ = normalize(thrust)
        x_world = np.array([np.cos(out.yaw), np.sin(out.yaw), 0])
        y_body, _ = normalize(self.__vcross(z_body, x_world))
        x_body = self.__vcross(y_body, z_body)

        project_unit = np.dot(jerk, z_body) * z_body
        jerk_orth_zbody = np.subtract(jerk, project_unit)
        scale = (1 / norm2(thrust))
        h_w = scale * jerk_orth_zbody 

        out.omega[0] = -1 * np.dot(h_w, y_body)
        out.omega[1] = np.dot(h_w, x_body)
        out.omega[2] = z_body[2] * dyaw

        return out




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    t = 0.2
    sim_time = 5
    duration = 1.05

    goal_state = [0.00004, -0.00002, 0.00002, 0]
    traj_gen = QuadTrajGen(poly_degree=7)
    current_eval = traj_eval()
    
    total_traj = [current_eval.pos]
    traj_gen.plan_go_to_from(current_eval=current_eval, desired_state=goal_state, duration=duration, current_time=t)
    
    # Get setpoint loops
    while(t < sim_time):  
        t += 0.1
        next_goal = traj_gen.piecewise_eval(t)
        total_traj.append(next_goal.pos)
        print(t, next_goal.pos, next_goal.vel)
        # if np.linalg.norm(test.pos-goal_state[:3]) < 0.5:
        #     break
        # current_state = [traj_gen.piecewise_eval(t).pos,  traj_gen.piecewise_eval(t).vel, traj_gen.piecewise_eval(t).acc, traj_gen.piecewise_eval(t).acc
    ax = plt.figure().add_subplot(projection='3d')
    for point in total_traj:
        ax.scatter(point[0], point[1], point[2])
    plt.show()
    # print(traj_gen.planner.planned_trajectory["pieces"][0])



    
