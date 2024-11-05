import math
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *
from scipy.spatial.transform import Rotation as R

GRAV = 9.81


class CollectiveThrustBodyRate(object):
    """
    Collective Thrust and Body Rate Controller. This implementation is intended to test the low level body rate controller.
    The low level body rate controller is implemented in quadrotor_dynamics.py
    NOTE: The controllers quaternion representation is [x,y,z,w] but the sim is [w,x,y,z]
    """
    def __init__(self, dynamics, dynamics_params):
        self.step_func = self.step
        self.controller_dynamics = dynamics
        self.controller_dynamics_params = dynamics_params
        
        self.J = np.diag(dynamics.inertia)
        
        self.ARCMINUTE = math.pi / 10800.0
        
        self.pwmToThrustA = 0.091492681
        self.pwmToThrustB = 0.067673604
        
        self.control_vector = np.zeros(4) # Thrust, BodyRate_X, BodyRate_Y, BodyRate_Z
        self.control_omega = np.zeros(3)
        
        self.tau_xy = 0.3 # 0.3
        self.zeta_xy = 0.95 # 0.85
        
        self.tau_z = 0.3
        self.zeta_z = 0.85
        
        self.tau_rp = 0.1 # 0.25
        self.mixing_factor = 1.0
        
        self.tau_rp_rate = 0.015
        self.tau_yaw_rate = 0.0075
        
        self.coll_min = 1
        self.coll_max = 18
        
        self.thrust_reduction_fairness = 0.25
        
        self.omega_rp_max = 35 #30
        self.omega_yaw_max = 35 #10
        self.heuristic_rp = 12
        self.heuristic_yaw = 5
        
        
    def step(self, dynamics, action, goal, dt, observation=None):
        
        def lin_transform(a, out_max, out_min):
            """
            Transforms value a into range [out_max, out_min] given input ranges [in_max, in_min]
            """
            in_min = -35
            in_max = 35


            return (a - in_min) * ((out_max - out_min)/(in_max - in_min)) + out_min
        
        accDes = np.zeros(3)
        
        collCmd = 0.0
        
        attErrorReduced = self.qeye() #Defaults to identity quaternion
        
        attErrorFull = self.qeye()
        
        attDesiredFull = self.qeye()
        
        r_temp = R.from_matrix(dynamics.rot)
        attitude = r_temp.as_quat() # Get current quad rotation as quaternion [x, y, z, w]
        
        attitudeI = self.qinv(attitude)
        
        R02 = (2.0 * attitude[0] * attitude[2]) + (2 * attitude[3] * attitude[1])
        R12 = (2.0 * attitude[1] * attitude[2]) - (2 * attitude[3] * attitude[0])
        R22 = (attitude[3] * attitude[3]) - (attitude[0] * attitude[0]) - (attitude[1] * attitude[1]) + (attitude[2] * attitude[2])
        
        # current_R = dynamics.rot
        # R02 = current_R[2][0]
        # R12 = current_R[2][1]
        # R22 = current_R[2][2]
        
        temp1 = self.qeye()
        temp2 = self.qeye()
        
        goal[3:] = np.zeros(10)
        
        pError = goal[:3] - dynamics.pos
        vError = goal[3:6] - dynamics.vel
        
        # Linear Control
        
        accDes[0] = 0.0
        accDes[0] += 1.0 / self.tau_xy / self.tau_xy * pError[0]
        accDes[0] += 2.0 * self.zeta_xy / self.tau_xy * vError[0]
        accDes[0] += goal[6]
        accDes[0] = self.constrain(accDes[0], -self.coll_max, self.coll_max)
        
        accDes[1] = 0.0
        accDes[1] += 1.0 / self.tau_xy / self.tau_xy * pError[1]
        accDes[1] += 2.0 * self.zeta_xy / self.tau_xy * vError[1]
        accDes[1] += goal[7]
        accDes[1] = self.constrain(accDes[1], -self.coll_max, self.coll_max)
        
        accDes[2] = dynamics.gravity
        accDes[2] += 1.0 / self.tau_z / self.tau_z * pError[2]
        accDes[2] += 2.0 * self.zeta_z / self.tau_z * vError[2]
        accDes[2] += goal[8]
        accDes[2] = self.constrain(accDes[2], -self.coll_max, self.coll_max)
        
        
        # Thrust Control
        collCmd = accDes[2] / R22
        if (abs(collCmd) > self.coll_max):
            x = accDes[0]
            y = accDes[1]
            z = accDes[2] - dynamics.gravity
            f = self.constrain(self.thrust_reduction_fairness, 0, 1)
            
            r = 0.0
            
            a = x**2 + y**2 + (z*f)**2
            if (a < 0):
                a = 0.0
                
            b = 2 * z*f*((1-f)*z + dynamics.gravity)
            c = self.coll_max**2 - ((1-f)*z + dynamics.gravity)**2
            
            if (c<0):
                c = 0.0
            if (abs(a)< 1e-6):
                r = 0.0
            else:
                sqrrterm = b**2 + 4.0*a*c
                r = (-b + math.sqrt(sqrrterm))/(2.0*a)
                r = self.constrain(r, 0, 1)
                
            accDes[0] = r*x
            accDes[1] = r*y
            accDes[2] = (r*f+(1-f))*z + dynamics.gravity
        
        collCmd = self.constrain(accDes[2] / R22, self.coll_min, self.coll_max) 
        
        zI_des = self.normalize(accDes)
        zI_cur = self.normalize(np.array([R02, R12, R22]))
        zI = np.array([0.0, 0.0, 1.0])
        
        # Reduced Attitude Control
        
        dotProd = self.vdot(zI_cur, zI_des)
        dotProd = self.constrain(dotProd, -1 , 1)
        
        alpha = np.arcsin(dotProd)
        
        rotAxisI = np.zeros(3)
        if (abs(alpha) > (1 * self.ARCMINUTE)):
            rotAxisI = self.normalize(self.vcross(zI_cur, zI_des))
        else:
            rotAxisI = np.array([1.0,1.0,0.0])

        attErrorReduced[3] = np.cos(alpha / 2.0)
        attErrorReduced[0] = np.sin(alpha / 2.0) * rotAxisI[0] 
        attErrorReduced[1] = np.sin(alpha / 2.0) * rotAxisI[1]
        attErrorReduced[2] = np.sin(alpha / 2.0) * rotAxisI[2]
        
        if (np.sin(alpha / 2.0)) < 0:
            rotAxisI = -1.0 * rotAxisI
        if (np.cos(alpha / 2.0)) < 0:
            rotAxisI = -1.0 * rotAxisI
            attErrorReduced = -1.0 * attErrorReduced

        attErrorReduced = self.qnormalize(attErrorReduced)
        
        # Full Attitude Control
        dotProd = self.vdot(zI, zI_des)
        dotProd = self.constrain(dotProd, -1.0, 1.0)
        alpha = np.arccos(dotProd)
        
        if (abs(alpha) > (1 * self.ARCMINUTE)):
            rotAxisI = self.normalize(self.vcross(zI, zI_des))
        else:
            rotAxisI = np.array([1.0, 1.0, 0.0])
       
        attFullReqPitchRoll = np.array([np.sin(alpha / 2.0) * rotAxisI[0], 
                                        np.sin(alpha / 2.0) * rotAxisI[1],
                                        np.sin(alpha / 2.0) * rotAxisI[2],
                                        np.cos(alpha / 2.0)])
        
        attFullReqYaw = np.array([0.0, 0.0, np.sin(np.radians(goal[12]) / 2.0), np.cos(np.radians(goal[12]) / 2.0)])

        attDesiredFull = self.qqmul(attFullReqPitchRoll, attFullReqYaw)
        
        attErrorFull = self.qqmul(attitudeI, attDesiredFull)
        
        if (attErrorFull[3] < 0):
            attErrorFull = -1.0 * attErrorFull
            attDesiredFull = self.qqmul(attitude, attErrorFull)
            
        attErrorFull = self.qnormalize(attErrorFull)
        attDesiredFull = self.qnormalize(attDesiredFull)
        
        
        # Mixing Full and Reduced Control
        attError = self.qeye()
        
        if (self.mixing_factor <= 0):
            attError = attErrorReduced
        elif (self.mixing_factor >= 1):
            attError = attErrorFull
        else:
            temp1 = self.qinv(attErrorReduced)
            temp2 = self.qnormalize(self.qqmul(temp1, attErrorFull))
        
            alpha = 2.0 * np.arccos(self.constrain(temp2[3], -1., 1.))
            
            temp1 = np.array([0., 0., 
                            np.sin(alpha * self.mixing_factor / 2.0) * (-1.0 if temp2[2] < 0 else 1.0),
                            np.cos(alpha * self.mixing_factor / 2.0)])
            attError = self.qnormalize(self.qqmul(attErrorReduced, temp1))
            
            
        # Control Signals
        self.control_omega[0] = 2.0 / self.tau_rp * attError[0]
        self.control_omega[1] = 2.0 / self.tau_rp * attError[1]
        self.control_omega[2] = 2.0 / self.tau_rp * attError[2] + goal[11]
        
        # if (((self.control_omega[0] * dynamics.omega[0]) < 0) and (abs(dynamics.omega[0]) > self.heuristic_rp)):
        #     if (dynamics.omega[0] < 0):
        #         sign = -1.0
        #     else:
        #         sign = 1.0
        #     self.control_omega[0] = self.omega_rp_max * sign
            
        # if (((self.control_omega[1] * dynamics.omega[1]) < 0) and (abs(dynamics.omega[1]) > self.heuristic_rp)):
        #     if (dynamics.omega[0] < 0):
        #         sign = -1.0
        #     else:
        #         sign = 1.0
        #     self.control_omega[1] = self.omega_rp_max * sign
            
        # if (((self.control_omega[2] * dynamics.omega[2]) < 0) and (abs(dynamics.omega[2]) > self.heuristic_yaw)):
        #     if (dynamics.omega[0] < 0):
        #         sign = -1.0
        #     else:
        #         sign = 1.0
        #     self.control_omega[2] = self.omega_rp_max * sign
        
        # scaling = 1
        # scaling = max(scaling, abs(self.control_omega[0]) / self.omega_rp_max)
        # scaling = max(scaling, abs(self.control_omega[1]) / self.omega_rp_max)
        # scaling = max(scaling, abs(self.control_omega[2]) / self.omega_yaw_max)
        
        # self.control_omega[0] /= scaling
        # self.control_omega[1] /= scaling
        # self.control_omega[2] /= scaling
        
        self.control_thrust = collCmd * dynamics.mass # Desired acc * mass = thrust (N)
        self.control_thrust = self.control_thrust / np.sum(dynamics.thrust_max)
        self.control_omega = lin_transform(self.control_omega, out_max=1, out_min=0)
        
        desired_state = np.array([self.control_thrust, self.control_omega[0], 
                                            self.control_omega[1], self.control_omega[2]])

        dynamics.step(desired_state, dt)
        self.action = desired_state.copy()

    def vcross(self, a, b):
        return np.array([(a[1]*b[2]) - (a[2]*b[1]), (a[2]*b[0]) - (a[0]*b[2]), (a[0]*b[1]) - (a[1]*b[0])])
        
    
    def vdot(self, a, b):
        return (a[0] * b[0]) + (a[1] * b[1]) + (a[2] * b[2])
    
    def mvmul(self, a, v):
        x = a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2]
        y = a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2]
        z = a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2]
        
        return np.array([x,y,z])
    def normalize(self, x):
        # n = norm(x)
        n = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5  # np.sqrt(np.cumsum(np.square(x)))[2]

        if n < 0.00001:
            return x, 0
        return (1/n)*x
        
    def constrain(self, a, minVal, maxVal):
        return min(maxVal, max(minVal, a))
        
    def qeye(self):
        return np.array([0., 0., 0., 1.])
    
    def qinv(self, quat):
        return np.array([-quat[0], -quat[1], -quat[2], quat[3]])
    
    def qnormalize(self, quat):
        # Normalize for precision
        s = 1.0 / math.sqrt(np.dot(quat,quat))
        return s * quat
    
    def qqmul(self, q, p):

        x = q[3]*p[0] + q[2]*p[1] - q[1]*p[2] + q[0]*p[3]
        y = -q[2]*p[0] + q[3]*p[1] + q[0]*p[2] + q[1]*p[3]
        z = q[1]*p[0] - q[0]*p[1] + q[3]*p[2] + q[2]*p[3]
        w = -q[0]*p[0] - q[1]*p[1] - q[2]*p[2] + q[3]*p[3]
        
        return np.array([x,y,z,w])
    
    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)
