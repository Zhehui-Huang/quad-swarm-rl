import numpy as np
import copy
from quad_utils import normalize, cross_vec, norm2
from quadrotor_planner import QuadPlanner, poly4d, traj_eval

class QuadTrajGen:
    """ 
        This class handles trajectory generation for a crazyflie quadrotor.
    """
    def __init__(
        self,
        poly_degree: int
    ):

        self.planner = QuadPlanner(poly_degree=7)
        self.poly_degree = 7
        self.stored_goal_points = []
        self.PI = 3.14159265358979323846

    def plan_go_to_from(
        self,
        initial_state: traj_eval,
        desired_state: "np.ndarray[np.float_]",
        duration: float,
        current_time: float 
    ):
        """ 
        Note: Python adaptation from plan_go_to_from() of the crazyflie firmware.
        Copyright (c) 2018 Wolfgang Hoenig and James Alan Preiss
        Modifcations by: Darren Chiu (chiudarr@usc.edu)
        
        Arguments:
            initial_state: [x, y, z, vx, vy, vz, roll, pitch, yaw, omega_x, omega_y, omega_z]
            desired_state: [x, y, z, yaw]
        """

        curr_yaw = self.__normalize_radians(initial_state.yaw)
        desired_yaw = self.__normalize_radians(desired_state[3])
        goal_yaw = curr_yaw + self.__smallest_signed_angle(curr_yaw, desired_yaw)

        hover_pos = np.array([desired_state[0], desired_state[1], desired_state[2]])

        self.__piecewise_plan_7th_order_no_jerk(duration, initial_state.pos, curr_yaw, initial_state.vel, 
        initial_state.omega[2], initial_state.acc, hover_pos, goal_yaw, np.zeros(3), 0, np.zeros(3))

        # self.planner.planned_trajectory["pieces"][0].poly[0] = np.array([0.000000, -0.000000, 0.000000, -0.000000, 0.830443, -0.276140, -0.384219, 0.180493])
        # self.planner.planned_trajectory["pieces"][0].poly[1] = np.array([-0.000000, 0.000000, -0.000000, 0.000000, -1.356107, 0.688430, 0.587426, -0.329106])
        # self.planner.planned_trajectory["pieces"][0].poly[2] = np.array([ 0, 0, 0, 0, 0, 0, 0, 0])
        # self.planner.planned_trajectory["pieces"][0].poly[3] = np.array([ 0, 0, 0, 0, 0, 0, 0, 0])

        self.planner.planned_trajectory["t_begin"] = current_time

    def piecewise_eval(self, t) -> traj_eval:
        """ Generates the next goal point based on current time t (seconds).
            Returns a goal point as a traj_eval class.
        """
        cursor = 0
        t = t - self.planner.planned_trajectory["t_begin"]
        while (cursor < self.planner.planned_trajectory["n_pieces"]):
            piece = self.planner.planned_trajectory["pieces"][cursor]

            if (t <= piece.duration):
                poly4d_tmp = copy.deepcopy(piece)

                return self.__poly4d_eval(poly4d_tmp, t)
            
            t = t - piece.duration
            cursor = cursor + 1

        end_piece = self.planner.planned_trajectory["pieces"][self.planner.planned_trajectory["n_pieces"] - 1]
        ev = self.__poly4d_eval(end_piece, end_piece.duration)
        ev.vel = np.zeros(3)
        ev.acc = np.zeros(3)
        ev.omega = np.zeros(3)
        
        return ev

    def __piecewise_plan_7th_order_no_jerk(self, duration, p0, y0, v0, dy0, a0, p1, y1, v1, dy1, a1):
        p = poly4d(self.poly_degree)
        p.duration = duration

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

    def __poly4d_derivative(self, piece: poly4d) -> poly4d:
        # Take derivative of each state
        for i in range(4):
            #Iterate through each exponent
            for j in range(1, piece.degree+1):
                print(j)
                piece.poly[i][j-1] = j * piece.poly[i][j]

            piece.poly[i][piece.degree] = 0
        
        return piece

    def __normalize_radians(self, angle: float) -> float:
        """ Normalize radians to the range: [-pi, pi]"""
        return np.arctan2(np.sin(angle),np.cos(angle))

    def __smallest_signed_angle(self, start, goal) -> float:
        diff = goal - start
        return (diff + self.PI) % (2*self.PI) - self.PI


    def __vcross(self, a, b):
        return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])
            
    def __poly4d_eval(self, piece: poly4d, t: float) -> traj_eval:
        # Output of desired states
        out = traj_eval()
        out.pos = self.__polyval_xyz(piece, t)
        out.yaw = self.__polyval_yaw(piece, t)

        # 1st Derivative
        piece = self.__poly4d_derivative(piece)
        out.vel = self.__polyval_xyz(piece, t)
        dyaw = self.__polyval_yaw(piece, t)

        # 2nd Derivative
        piece = self.__poly4d_derivative(piece)
        out.acc = self.__polyval_xyz(piece, t)

        # 3rd Derivative
        piece = self.__poly4d_derivative(piece)
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

        self.stored_goal_points.append(out)

        return out




if __name__ == "__main__":

    """ Example use case for trajectory generation class."""

    import matplotlib.pyplot as plt

    # Create trajectory generator class
    traj_gen = QuadTrajGen(poly_degree=7)

    # Begining time of episode in seconds
    t = 0.0 
    # Duration needed to complete trajectory in seconds
    duration = 10.05

    # Create an instance goal point for the initial state. 
    initial_state = traj_eval()
    initial_state.set_initial_state([0,0,0.65,0])
    
    #Desired FINAL goal point
    goal_state = [5.00000, 1.00000, 0.65, 0]

    print("Initial State:", initial_state)

    total_traj = [initial_state]
    traj_gen.plan_go_to_from(initial_state=initial_state, desired_state=goal_state, duration=duration, current_time=t)
    print(traj_gen.planner.planned_trajectory["pieces"])

    sim_time = 15

    # Get setpoint loops. This can be understood as the simulation steps.
    while(t < sim_time):  

        t += 0.1

        next_goal = traj_gen.piecewise_eval(t)
        total_traj.append(next_goal)

    print("Final State: ", total_traj[-1])

    # ax = plt.figure().add_subplot(projection='3d')
    # for point in total_traj:
    #     ax.scatter(point.pos[0], point.pos[1], point.pos[2], c='b')


    plt.figure()
    vx = []
    vy = []
    vz = []
    for point in total_traj:
        vx.append(point.vel[0])
        vy.append(point.vel[1])
        vz.append(point.vel[2])
    plt.plot(vx)
    plt.plot(vy)
    plt.plot(vz)

    plt.show()
    
    # print(traj_gen.planner.planned_trajectory["pieces"][0])



    
