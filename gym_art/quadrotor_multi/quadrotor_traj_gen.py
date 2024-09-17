import numpy as np
import copy
from gym_art.quadrotor_multi.quad_utils import normalize, cross_vec, norm2
from gym_art.quadrotor_multi.quadrotor_planner import QuadPlanner, poly4d, traj_eval

class QuadTrajGen:
    """ 
        This class handles trajectory generation for a crazyflie quadrotor.
    """
    def __init__(self, poly_degree: int):
        self.planner = QuadPlanner(poly_degree=7)
        self.poly_degree = 7
        self.stored_goal_points = []
        self.PI = 3.14159265358979323846
        self.start_point = np.zeros(3)
        self.end_point = np.zeros(4)

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
        Modifcations by: Darren Chiu 
        
        Arguments:
            initial_state: [x, y, z, vx, vy, vz, roll, pitch, yaw, omega_x, omega_y, omega_z]
            desired_state: [x, y, z, yaw]
        """
        self.start_point = initial_state.as_nparray()[0:3]
        self.end_point = desired_state

        curr_yaw = self.__normalize_radians(initial_state.yaw)
        desired_yaw = self.__normalize_radians(desired_state[3])
        goal_yaw = curr_yaw + self.__smallest_signed_angle(curr_yaw, desired_yaw)

        hover_pos = np.array([desired_state[0], desired_state[1], desired_state[2]])

        self.__piecewise_plan_7th_order_no_jerk(duration, initial_state.pos, curr_yaw, initial_state.vel, 
        initial_state.omega[2], initial_state.acc, hover_pos, goal_yaw, np.zeros(3), 0, np.zeros(3))

        self.planner.planned_trajectory["t_begin"] = current_time
        # print("PLANNER: Start", initial_state)
        # print("PLANNER: End", desired_state)

    def get_initial_state(self):
        return self.start_point

    def get_final_state(self):
        return self.end_point

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

        end_piece = copy.deepcopy(self.planner.planned_trajectory["pieces"][-1])
        ev = self.__poly4d_eval(end_piece, end_piece.duration)
        ev.vel = np.zeros(3)
        ev.acc = np.zeros(3)
        ev.omega = np.zeros(3)
        
        return ev

    def get_poly(self):
        return self.planner.planned_trajectory["pieces"][0].poly

    def plan_finished(self, t) -> int:
        """ Returns if the current goal point should have been reached given the current time. """
        
        return int((t - self.planner.planned_trajectory["t_begin"]) >= self.planner.planned_trajectory["pieces"][0].duration)

    def get_trajectory(self) -> list:
        """ Returns the most previously evaluated trajectory """
        return self.stored_goal_points   

    def __piecewise_plan_7th_order_no_jerk(self, duration, p0, y0, v0, dy0, a0, p1, y1, v1, dy1, a1):
        p = poly4d(degree=self.poly_degree)
        p.duration = duration

        self.planner.planned_trajectory["n_pieces"] = 1

        self.__poly7_nojerk(p.poly[0], duration, p0[0], v0[0], a0[0], p1[0], v1[0], a1[0])
        self.__poly7_nojerk(p.poly[1], duration, p0[1], v0[1], a0[1], p1[1], v1[1], a1[1])
        self.__poly7_nojerk(p.poly[2], duration, p0[2], v0[2], a0[2], p1[2], v1[2], a1[2])
        self.__poly7_nojerk(p.poly[3], duration, y0, dy0, 0, y1, dy1, 0)

        # self.planner.planned_trajectory["pieces"].append(p)
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
        
        if (abs(x) < 0.001):
            x = 0

        return x

    def __poly4d_derivative(self, piece: poly4d):
        # Take derivative of each state
        for i in range(4):
            #Iterate through each exponent
            for j in range(1, piece.degree+1):
                piece.poly[i][j-1] = j * piece.poly[i][j]

            piece.poly[i][piece.degree] = 0

    def __normalize_radians(self, angle: float) -> float:
        """ Normalize radians to the range: [-pi, pi]"""
        return np.arctan2(np.sin(angle),np.cos(angle))

    def __smallest_signed_angle(self, start, goal) -> float:
        diff = goal - start
        return (diff + self.PI) % (2*self.PI) - self.PI

    def __vprojectunit(self, a, b_unit):
        return np.dot(a, b_unit) * b_unit

    def __vorthunit(self, a, b_unit):
        return np.subtract(a, self.__vprojectunit(a, b_unit))

    def __vcross(self, a, b):
        return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])

    def __mag(self, a):
        return (np.dot(a, a) ** 0.5)
            
    def __poly4d_eval(self, piece: poly4d, t: float) -> traj_eval:
        # Output of desired states
        out = traj_eval()
        out.pos = self.__polyval_xyz(piece, t)
        out.yaw = self.__polyval_yaw(piece, t)
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

        # project_unit = np.dot(jerk, z_body) * z_body
        # jerk_orth_zbody = np.subtract(jerk, project_unit)
        jerk_orth_zbody = self.__vorthunit(jerk, z_body)
        scale = (1.0 / self.__mag(thrust))
        h_w = scale * jerk_orth_zbody 

        roll_rate = -1 * np.dot(h_w, y_body)
        pitch_rate = np.dot(h_w, x_body)
        yaw_rate = z_body[2] * dyaw

        out.omega = np.array([roll_rate, pitch_rate, yaw_rate])

        return out    



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import unittest

    class TestTrajGen(unittest.TestCase):

        np.random.seed(3333)

        def test_end_points(self):
            sim_time = 15
            num_agents = 15
            ROOM_DIM = 10

            #Conduct test multiple times
            for k in range(100):
                # goal_generator = []
                goal_generator = [QuadTrajGen(poly_degree=7) for i in range(num_agents)]
                # start_point = []
                start_point = [np.zeros(3) for i in range(num_agents)]
                # end_point = []
                end_point = [np.zeros(3) for i in range(num_agents)]

                yaw = [0 for i in range(num_agents)]

                
                for i in range(num_agents):
                    initial_state = traj_eval()
                    start_point[i] = (np.random.uniform(low=-ROOM_DIM, high=ROOM_DIM, size=(3,)))
                    initial_state.set_initial_pos(start_point[i])

                    end_point[i] = (np.random.uniform(low=-ROOM_DIM, high=ROOM_DIM, size=(3,)))
                    yaw[i] = np.random.uniform(low=-3.14, high=3.14)

                    end_point[i] = np.append(end_point[i], yaw[i])
                    
                    dist = np.linalg.norm(start_point[i][0:3] - end_point[i][0:3])
                    
                    traj_duration = np.random.uniform(low=dist/2, high=sim_time-2)
                    
                    goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=end_point[i], 
                                                        duration=traj_duration, current_time=0)
                
                for i in range(num_agents):
                    A = goal_generator[i].piecewise_eval(20).as_nparray()
                    B = end_point[i]
                    
                    self.assertTrue(np.linalg.norm(A[0:3]-B[0:3]) < 0.1, "End point positions do not match: {}, {}".format(A[0:3],B[0:3]))
                    self.assertTrue(abs(A[12] - B[3]) < 0.01, "End Point Yaws do not match: {}, {}".format(A[12], B[3]))
            
        def test_same_goal(self):
            sim_time = 15

            ROOM_DIM = 10
            max_room_dist = np.linalg.norm([ROOM_DIM, ROOM_DIM, ROOM_DIM])

            max_vel = 2.0 
            max_omega = 35

            num_agents = 10

            traj_gen = QuadTrajGen(poly_degree=7)

            for i in range(5):
                initial_state = traj_eval()

                initial_state.set_initial_pos(np.random.uniform(low=-ROOM_DIM, high=ROOM_DIM, size=(3,)))
                final_goal = initial_state.pos

                dist = np.linalg.norm(initial_state.pos - final_goal)
                
                traj_duration = np.random.uniform(low = dist, high=sim_time-1)

                traj_gen.plan_go_to_from(initial_state=initial_state, desired_state=np.append(final_goal, np.random.uniform(low=0, high=3.14)), 
                                                    duration=traj_duration, current_time=0)
                
                # Simulation Loop
                t = 0
                while (t < sim_time):
                    next_goal = traj_gen.piecewise_eval(t)
                    self.assertLess(np.linalg.norm(next_goal.pos), max_room_dist, "Failed Goal Position Boundary")
                    
                    t += 0.05

        def test_random_goal(self):
            sim_time = 15

            ROOM_DIM = 10
            max_room_dist = np.linalg.norm([ROOM_DIM, ROOM_DIM, ROOM_DIM])

            max_vel = 2.0 
            max_omega = 35

            num_agents = 10

            end_points = []
            goal_generator = []

            for i in range(100):
                goal_generator.append(QuadTrajGen(poly_degree=7))

            for i in range(100):
                initial_state = traj_eval()

                initial_state.set_initial_pos(np.random.uniform(low=-ROOM_DIM, high=ROOM_DIM, size=(3,)))
                final_goal = np.random.uniform(low=-ROOM_DIM, high=ROOM_DIM, size=(3,))

                final_goal[2] = 0.65

                dist = np.linalg.norm(initial_state.pos - final_goal)
                
                traj_duration = np.random.uniform(low = dist , high=sim_time-1)

                goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=np.append(final_goal, np.random.uniform(low=0, high=3.14)), 
                                                    duration=traj_duration, current_time=0)
                
                next_goal = goal_generator[i].piecewise_eval(0)
                next_goal = next_goal.as_nparray()
                end_points.append(next_goal)
                for j in range(3):
                        self.assertTrue(-ROOM_DIM <= next_goal[j] <= ROOM_DIM)

                # Simulation Loop
                t = 0
                while (t < sim_time):
                    next_goal = goal_generator[i].piecewise_eval(t)
                    next_goal = next_goal.as_nparray()
                    # print(next_goal.pos)
                    for j in range(3):
                        self.assertTrue(-ROOM_DIM <= next_goal[j] <= ROOM_DIM, "Failed Random Goal Test")
                    
                    t += 0.05
        
    unittest.main(verbosity=2)

    """ Example use case for trajectory generation class."""

    # Create trajectory generator class
    traj_gen = QuadTrajGen(poly_degree=7)

    # Begining time of episode in seconds
    t = 0.0 
    # Duration needed to complete trajectory in seconds
    duration = 5.0

    # # Create an instance goal point for the initial state. 
    # initial_state = traj_eval()
    # initial_state.set_initial_pos([1.25,-0.75,2.138])
    # initial_state.set_initial_yaw(0)
    
    # #Desired FINAL goal point
    # goal_state = [2.75, -2.25, 2.617, 0]

    # Create an instance goal point for the initial state. 
    initial_state = traj_eval()
    initial_state.set_initial_pos([0.9,-3.2,0.2])
    initial_state.set_initial_yaw(3.14/2)
    
    #Desired FINAL goal point
    goal_state = [-1.2, -2.25, 2.617, 0]

    # print("Initial State:", initial_state)

    total_traj = [initial_state]
    traj_gen.plan_go_to_from(initial_state=initial_state, desired_state=goal_state, duration=duration, current_time=t)
    
    sim_time = 20
    # print("Initial Goal:", traj_gen.piecewise_eval(0.001))

    # Get setpoint loops. This can be understood as the simulation steps.
    while(t < sim_time):  

        t += 0.005
        # Evaluate the next goal point based on the current simulation time (in seconds).
        next_goal = traj_gen.piecewise_eval(t)

        # # We can also replan during the simulation. 
        # # NOTE: THIS SHOULD ONLY BE CALLED AFTER THE PREVIOUS PLAN HAS FINISHED.
        if (traj_gen.plan_finished(t)):
            traj_gen.plan_go_to_from(initial_state=next_goal, desired_state=[0, 0, 0.65, 0], duration=duration, current_time=t)
            

    print("Final State: ", total_traj[-1])


    fig, axs = plt.subplots(3)

    x = []
    y = []
    z = []
    vx = []
    vy = []
    vz = []

    o_r = []
    o_p = []
    o_y = []

    yaw = []
    for point in traj_gen.get_trajectory():
        x.append(point.pos[0])
        y.append(point.pos[1])
        z.append(point.pos[2])
        vx.append(point.vel[0])
        vy.append(point.vel[1])
        vz.append(point.vel[2])
        o_r.append(point.omega[0])
        o_p.append(point.omega[1])
        o_y.append(point.omega[2])
        yaw.append(point.yaw)
    axs[0].plot(x, label="x")
    axs[0].plot(y, label="y")
    axs[0].plot(z, label="z")
    axs[0].plot(yaw, label='yaw')
    axs[0].legend()
    axs[1].plot(vx)
    axs[1].plot(vy)
    axs[1].plot(vz)

    # axs[2].set_ylim(-3.14, 3.14)
    axs[2].plot(o_r)
    axs[2].plot(o_p)
    axs[2].plot(o_y)

    plt.show()
    
    # print(traj_gen.planner.planned_trajectory["pieces"][0])
