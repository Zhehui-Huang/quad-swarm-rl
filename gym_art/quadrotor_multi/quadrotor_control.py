from numpy.linalg import norm, inv
import math
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *
from scipy.spatial.transform import Rotation as R

GRAV = 9.81


class CollectiveThrustBodyRate(object):
    """
    Collective Thrust and Body Rate Controller.
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
        attErrorReduced[0] = np.sin(alpha / 2.0) * rotAxisI[0] #unstable behavior here
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
        
        if (((self.control_omega[0] * dynamics.omega[0]) < 0) and (abs(dynamics.omega[0]) > self.heuristic_rp)):
            if (dynamics.omega[0] < 0):
                sign = -1.0
            else:
                sign = 1.0
            self.control_omega[0] = self.omega_rp_max * sign
            
        if (((self.control_omega[1] * dynamics.omega[1]) < 0) and (abs(dynamics.omega[1]) > self.heuristic_rp)):
            if (dynamics.omega[0] < 0):
                sign = -1.0
            else:
                sign = 1.0
            self.control_omega[1] = self.omega_rp_max * sign
            
        if (((self.control_omega[2] * dynamics.omega[2]) < 0) and (abs(dynamics.omega[2]) > self.heuristic_yaw)):
            if (dynamics.omega[0] < 0):
                sign = -1.0
            else:
                sign = 1.0
            self.control_omega[2] = self.omega_rp_max * sign
        
        scaling = 1
        scaling = max(scaling, abs(self.control_omega[0]) / self.omega_rp_max)
        scaling = max(scaling, abs(self.control_omega[1]) / self.omega_rp_max)
        scaling = max(scaling, abs(self.control_omega[2]) / self.omega_yaw_max)
        
        self.control_omega[0] /= scaling
        self.control_omega[1] /= scaling
        self.control_omega[2] /= scaling
        
        self.control_thrust = collCmd
        
        desired_state = np.array([self.control_thrust, self.control_omega[0], 
                                            self.control_omega[1], self.control_omega[2]])
        
        dynamics.step(desired_state, dt)
        self.action = desired_state.copy()

    def vcross(self, a, b):
        return np.array([(a[1]*b[2]) - (a[2]*b[1]), (a[2]*b[0]) - (a[0]*b[2]), (a[0]*b[1]) - (a[1]*b[0])])
        
        
    # This function might not be necessary during Simulation?
    def compute_desired_pwm(self, desired_thrusts):
        for i in range(4):
            desired_thrust = desired_thrusts[i]
            motor_pwm = (-self.pwmToThrustB + math.sqrt(self.pwmToThrustB * self.pwmToThrustB + 4.0 * self.pwmToThrustA * desired_thrust)) / (2.0 * self.pwmToThrustA)
            normalized_actions[i] = motor_pwm
            # actions[i] = action

        normalized_actions = np.clip(normalized_actions, a_min=-np.ones(4), a_max=np.ones(4))
        normalized_actions = 0.5* (normalized_actions + 1.0)
        print(normalized_actions)
    
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
# import line_profiler
# like raw motor control, but shifted such that a zero action
# corresponds to the amount of thrust needed to hover.
class ShiftedMotorControl(object):
    def __init__(self, dynamics):
        pass

    def action_space(self, dynamics):
        # make it so the zero action corresponds to hovering
        low = -1.0 * np.ones(4)
        high = (dynamics.thrust_to_weight - 1.0) * np.ones(4)
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        action = (action + 1.0) / dynamics.thrust_to_weight
        action[action < 0] = 0
        action[action > 1] = 1
        dynamics.step(action, dt)


class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        # print("RawControl: self.zero_action_middle", self.zero_action_middle)
        self.action = None
        self.step_func = self.step

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
    def step(self, dynamics, action, goal, dt, observation=None):
        # action = np.clip(action, a_min=self.low, a_max=self.high)
        # action = self.scale * (action + self.bias)      
        dynamics.step(action, dt)
        self.action = action.copy()

    # @profile
    def step_tf(self, dynamics, action, goal, dt, observation=None):
        # print('bias/scale: ', self.scale, self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()


class VerticalControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.step = self.step1D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(1)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(1)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(1)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]] * 4), dt)

    # modifies the dynamics in place.
    # @profile
    def step1D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0]]), dt)


class VertPlaneControl(object):
    def __init__(self, dynamics, zero_action_middle=True, dim_mode="3D"):
        self.zero_action_middle = zero_action_middle

        self.dim_mode = dim_mode
        if self.dim_mode == '2D':
            self.step = self.step2D
        elif self.dim_mode == '3D':
            self.step = self.step3D
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        self.step_func = self.step

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(2)
            self.bias = 0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(2)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(2)
        return spaces.Box(self.low, self.high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step3D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array([action[0], action[0], action[1], action[1]]), dt)

    # modifies the dynamics in place.
    # @profile
    def step2D(self, dynamics, action, goal, dt, observation=None):
        # print('action: ', action)
        action = self.scale * (action + self.bias)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        dynamics.step(np.array(action), dt)


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


# P-only linear controller on angular velocity.
# direct (ignoring motor lag) control of thrust magnitude.
class OmegaThrustControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, action, dt):
        kp = 5.0  # could be more aggressive
        omega_err = dynamics.omega - action[1:]
        dw_des = -kp * omega_err
        acc_des = GRAV * (action[0] + 1.0)
        des = np.append(acc_des, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1
        dynamics.step(thrusts, dt)


# TODO: this has not been tested well yet.
class VelocityYawControl(object):
    def __init__(self, dynamics):
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)

    def action_space(self, dynamics):
        vmax = 20.0  # meters / sec
        dymax = 4 * np.pi  # radians / sec
        high = np.array([vmax, vmax, vmax, dymax])
        return spaces.Box(-high, high, dtype=np.float32)

    # @profile
    def step(self, dynamics, action, dt):
        # needs to be much bigger than in normal controller
        # so the random initial actions in RL create some signal
        kp_v = 5.0
        kp_a, kd_a = 100.0, 50.0

        e_v = dynamics.vel - action[:3]
        acc_des = -kp_v * e_v + npa(0, 0, GRAV)

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        R = dynamics.rot
        zb_des, _ = normalize(acc_des)
        yb_des, _ = normalize(cross(zb_des, R[:, 0]))
        xb_des = cross(yb_des, zb_des)
        R_des = np.column_stack((xb_des, yb_des, zb_des))

        def vee(R):
            return np.array([R[2, 1], R[0, 2], R[1, 0]])

        e_R = 0.5 * vee(np.matmul(R_des.T, R) - np.matmul(R.T, R_des))
        omega_des = np.array([0, 0, action[3]])
        e_w = dynamics.omega - omega_des

        dw_des = -kp_a * e_R - kd_a * e_w
        # we want this acceleration, but we can only accelerate in one direction!
        # thrust_mag = np.dot(acc_des, dynamics.rot[:,2])
        thrust_mag = get_blas_funcs("thrust_mag", [acc_des, dynamics.rot[:, 2]])

        des = np.append(thrust_mag, dw_des)
        thrusts = np.matmul(self.Jinv, des)
        thrusts = np.clip(thrusts, a_min=0.0, a_max=1.0)
        dynamics.step(thrusts, dt)


# this is an "oracle" policy to drive the quadrotor towards a goal
# using the controller from Mellinger et al. 2011
class NonlinearPositionController(object):
    # @profile
    def __init__(self, dynamics, tf_control=True):
        # import tensorflow as tf
        jacobian = quadrotor_jacobian(dynamics)
        self.Jinv = np.linalg.inv(jacobian)
        ## Jacobian inverse for our quadrotor
        # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
        #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
        #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
        #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])
        self.action = None

        self.kp_p, self.kd_p = 4.5, 3.5
        self.kp_a, self.kd_a = 200.0, 50.0

        self.rot_des = np.eye(3)

        self.tf_control = False
        if tf_control:
            self.step_func = self.step_tf
            self.sess = tf.Session()
            self.thrusts_tf = self.step_graph_construct(Jinv_=self.Jinv, observation_provided=True)
            self.sess.run(tf.global_variables_initializer())
        else:
            self.step_func = self.step

    # modifies the dynamics in place.
    # @profile
    def step(self, dynamics, goal, dt, action=None, observation=None):
        to_goal = goal[0:3] - dynamics.pos
        # goal_dist = np.sqrt(np.cumsum(np.square(to_goal)))[2]
        goal_dist = (to_goal[0] ** 2 + to_goal[1] ** 2 + to_goal[2] ** 2) ** 0.5
        ##goal_dist = norm(to_goal)
        e_p = -clamp_norm(to_goal, 4.0)
        e_v = dynamics.vel
        # print('Mellinger: ', e_p, e_v, type(e_p), type(e_v))
        acc_des = -self.kp_p * e_p - self.kd_p * e_v + np.array([0, 0, GRAV])

        # I don't need to control yaw
        # if goal_dist > 2.0 * dynamics.arm:
        #     # point towards goal
        #     xc_des = to_xyhat(to_goal)
        # else:
        #     # keep current
        #     xc_des = to_xyhat(dynamics.rot[:,0])

        xc_des = self.rot_des[:, 0]
        # xc_des = np.array([1.0, 0.0, 0.0])

        # rotation towards the ideal thrust direction
        # see Mellinger and Kumar 2011
        zb_des, _ = normalize(acc_des)
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
        thrust_mag = np.dot(acc_des, R[:, 2])

        des = np.append(thrust_mag, dw_des)

        # print('Jinv:', self.Jinv)
        thrusts = np.matmul(self.Jinv, des)
        thrusts[thrusts < 0] = 0
        thrusts[thrusts > 1] = 1

        dynamics.step(thrusts, dt)
        self.action = thrusts.copy()

    def step_tf(self, dynamics, goal, dt, action=None, observation=None):
        # print('step tf')
        if not self.observation_provided:
            xyz = np.expand_dims(dynamics.pos.astype(np.float32), axis=0)
            Vxyz = np.expand_dims(dynamics.vel.astype(np.float32), axis=0)
            Omega = np.expand_dims(dynamics.omega.astype(np.float32), axis=0)
            R = np.expand_dims(dynamics.rot.astype(np.float32), axis=0)
            # print('step_tf: goal type: ', type(goal), goal[:3])
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)

            result = self.sess.run([self.thrusts_tf], feed_dict={self.xyz_tf: xyz,
                                                                 self.Vxyz_tf: Vxyz,
                                                                 self.Omega_tf: Omega,
                                                                 self.R_tf: R,
                                                                 self.goal_xyz_tf: goal_xyz})

        else:
            print('obs fed: ', observation)
            goal_xyz = np.expand_dims(goal[:3].astype(np.float32), axis=0)
            result = self.sess.run([self.thrusts_tf], feed_dict={self.observation: observation,
                                                                 self.goal_xyz_tf: goal_xyz})
        self.action = result[0].squeeze()
        dynamics.step(self.action, dt)

    def step_graph_construct(self, Jinv_=None, observation_provided=False):
        # import tensorflow as tf
        self.observation_provided = observation_provided
        with tf.variable_scope('MellingerControl'):

            if not observation_provided:
                # Here we will provide all components independently
                self.xyz_tf = tf.placeholder(name='xyz', dtype=tf.float32, shape=(None, 3))
                self.Vxyz_tf = tf.placeholder(name='Vxyz', dtype=tf.float32, shape=(None, 3))
                self.Omega_tf = tf.placeholder(name='Omega', dtype=tf.float32, shape=(None, 3))
                self.R_tf = tf.placeholder(name='R', dtype=tf.float32, shape=(None, 3, 3))
            else:
                # Here we will provide observations directly and split them
                self.observation = tf.placeholder(name='obs', dtype=tf.float32, shape=(None, 3 + 3 + 9 + 3))
                self.xyz_tf, self.Vxyz_tf, self.R_flat, self.Omega_tf = tf.split(self.observation, [3, 3, 9, 3], axis=1)
                self.R_tf = tf.reshape(self.R_flat, shape=[-1, 3, 3], name='R')

            R = self.R_tf
            # R_flat = tf.placeholder(name='R_flat', type=tf.float32, shape=(None, 9))
            # R = tf.reshape(R_flat, shape=(-1, 3, 3), name='R')

            # GOAL = [x,y,z, Vx, Vy, Vz]
            self.goal_xyz_tf = tf.placeholder(name='goal_xyz', dtype=tf.float32, shape=(None, 3))
            # goal_Vxyz = tf.placeholder(name='goal_Vxyz', type=tf.float32, shape=(None, 3))

            # Learnable gains with static initialization
            kp_p = tf.get_variable('kp_p', shape=[], initializer=tf.constant_initializer(4.5), trainable=True)  # 4.5
            kd_p = tf.get_variable('kd_p', shape=[], initializer=tf.constant_initializer(3.5), trainable=True)  # 3.5
            kp_a = tf.get_variable('kp_a', shape=[], initializer=tf.constant_initializer(200.0), trainable=True)  # 200.
            kd_a = tf.get_variable('kd_a', shape=[], initializer=tf.constant_initializer(50.0), trainable=True)  # 50.

            ## IN case you want to optimize them from random values
            # kp_p = tf.get_variable('kp_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 4.5
            # kd_p = tf.get_variable('kd_p', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=10.0), trainable=True)  # 3.5
            # kp_a = tf.get_variable('kp_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 200.
            # kd_a = tf.get_variable('kd_a', initializer=tf.random_uniform(shape=[1], minval=0.0, maxval=100.0), trainable=True)  # 50.

            to_goal = self.goal_xyz_tf - self.xyz_tf
            e_p = -tf.clip_by_norm(to_goal, 4.0, name='e_p')
            e_v = self.Vxyz_tf
            acc_des = -kp_p * e_p - kd_p * e_v + tf.constant([0, 0, 9.81], name='GRAV')
            print('acc_des shape: ', acc_des.get_shape().as_list())

            def project_xy(x, name='project_xy'):
                # print('x_shape:', x.get_shape().as_list())
                # x = tf.squeeze(x, axis=2)
                return tf.multiply(x, tf.constant([1., 1., 0.]), name=name)

            # goal_dist = tf.norm(to_goal, name='goal_xyz_dist')
            xc_des = project_xy(tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2), name='xc_des')
            print('xc_des shape: ', xc_des.get_shape().as_list())
            # xc_des = project_xy(R[:, 0])

            # rotation towards the ideal thrust direction
            # see Mellinger and Kumar 2011
            zb_des = tf.nn.l2_normalize(acc_des, axis=1, name='zb_dex')
            yb_des = tf.nn.l2_normalize(tf.cross(zb_des, xc_des), axis=1, name='yb_des')
            xb_des = tf.cross(yb_des, zb_des, name='xb_des')
            R_des = tf.stack([xb_des, yb_des, zb_des], axis=2, name='R_des')

            print('zb_des shape: ', zb_des.get_shape().as_list())
            print('yb_des shape: ', yb_des.get_shape().as_list())
            print('xb_des shape: ', xb_des.get_shape().as_list())
            print('R_des shape: ', R_des.get_shape().as_list())

            def transpose(x):
                return tf.transpose(x, perm=[0, 2, 1])

            # Rotational difference
            Rdiff = tf.matmul(transpose(R_des), R) - tf.matmul(transpose(R), R_des, name='Rdiff')
            print('Rdiff shape: ', Rdiff.get_shape().as_list())

            def tf_vee(R, name='vee'):
                return tf.squeeze(tf.stack([
                    tf.squeeze(tf.slice(R, [0, 2, 1], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 0, 2], [-1, 1, 1]), axis=2),
                    tf.squeeze(tf.slice(R, [0, 1, 0], [-1, 1, 1]), axis=2)], axis=1, name=name), axis=2)

            # def vee(R):
            #     return np.array([R[2, 1], R[0, 2], R[1, 0]])

            e_R = 0.5 * tf_vee(Rdiff, name='e_R')
            print('e_R shape: ', e_R.get_shape().as_list())
            # e_R[2] *= 0.2  # slow down yaw dynamics
            e_w = self.Omega_tf

            # Control orientation
            dw_des = -kp_a * e_R - kd_a * e_w
            print('dw_des shape: ', dw_des.get_shape().as_list())

            # we want this acceleration, but we can only accelerate in one direction!
            # thrust_mag = np.dot(acc_des, R[:, 2])
            acc_cur = tf.squeeze(tf.slice(R, begin=[0, 0, 2], size=[-1, 3, 1]), axis=2)
            print('acc_cur shape: ', acc_cur.get_shape().as_list())

            acc_dot = tf.multiply(acc_des, acc_cur)
            print('acc_dot shape: ', acc_dot.get_shape().as_list())

            thrust_mag = tf.reduce_sum(acc_dot, axis=1, keepdims=True, name='thrust_mag')
            print('thrust_mag shape: ', thrust_mag.get_shape().as_list())

            # des = np.append(thrust_mag, dw_des)
            des = tf.concat([thrust_mag, dw_des], axis=1, name='des')
            print('des shape: ', des.get_shape().as_list())

            if Jinv_ is None:
                # Learn the jacobian inverse
                Jinv = tf.get_variable('Jinv', initializer=tf.random_normal(shape=[4, 4], mean=0.0, stddev=0.1),
                                       trainable=True)
            else:
                # Jacobian inverse is provided
                Jinv = tf.constant(Jinv_.astype(np.float32), name='Jinv')
                # Jinv = tf.get_variable('Jinv', shape=[4,4], initializer=tf.constant_initializer())

            print('Jinv shape: ', Jinv.get_shape().as_list())
            ## Jacobian inverse for our quadrotor
            # Jinv = np.array([[0.0509684, 0.0043685, -0.0043685, 0.02038736],
            #                 [0.0509684, -0.0043685, -0.0043685, -0.02038736],
            #                 [0.0509684, -0.0043685,  0.0043685,  0.02038736],
            #                 [0.0509684,  0.0043685,  0.0043685, -0.02038736]])

            # thrusts = np.matmul(self.Jinv, des)
            thrusts = tf.matmul(des, tf.transpose(Jinv), name='thrust')
            thrusts = tf.clip_by_value(thrusts, clip_value_min=0.0, clip_value_max=1.0, name='thrust_clipped')
            return thrusts

    def action_space(self, dynamics):
        circle_per_sec = 2 * np.pi
        max_rp = 5 * circle_per_sec
        max_yaw = 1 * circle_per_sec
        min_g = -1.0
        max_g = dynamics.thrust_to_weight - 1.0
        low = np.array([min_g, -max_rp, -max_rp, -max_yaw])
        high = np.array([max_g, max_rp, max_rp, max_yaw])
        return spaces.Box(low, high, dtype=np.float32)

# TODO:
# class AttitudeControl,
# refactor common parts of VelocityYaw and NonlinearPosition
