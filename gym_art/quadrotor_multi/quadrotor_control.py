from numpy.linalg import norm, inv
import math
from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *

GRAV = 9.81


class CollectiveThrustBodyRate(object):
    """
    Collective Thrust and Body Rate Controller.
    """
    def __init__(self, dynamics, dynamics_params):
        self.controller_dynamics = dynamics
        self.controller_dynamics_params = dynamics_params
        
        self.naive = True # Set true if not using feed forward.
        self.use_NN = True # Set to true if we want to use a CT and BR generated from NN.
        
        self.J = np.diag(self.controller_dynamics.inertia)
        
        self.current_goal = None
        self.current_thrust = np.zeros(4)
        self.thrust_max = self.controller_dynamics.thrust_max # units of N
        
        # self.A_dynamics = np.array([[c*l, -c*l, -c*l, c*l],
        #             [-c*l, -c*l, c*l, c*l],
        #             [k, k, k, k]])
        
        # LQR Gains
        self.kwxy = 0.001
        self.kwxy = 0.001
        self.kwz = 0.001
        
        self.knxy = 0.001
        self.kxy = 0.001
        self.knz = 0.001
        
        # LQR Gain Matrix
        self.K_lqr =np.array([[self.kwxy, 0, 0, self.knxy, 0, 0],
                         [0, self.kwxy, 0, 0, self.knxy, 0],
                         [0, 0, self.kwz, 0, 0, self.knz]])
        
        self.w_des = np.array([0.0, 0.0, 0.0]) # N
        
    def compute_thrust_mix(self, goal):
        """
        Compute the individual rotor thrusts given desired body torque and collective thrust. 
        """
        self.current_goal = goal
        
        torque_desired = self.compute_desired_torque()
        
        # Collective thrust = Z_body * (acc + g * Z_world)

        c_desired = np.dot(self.controller_dynamics.rot[:, 2], (self.controller_dynamics.accelerometer))
        # np.array([0., 0., self.controller_dynamics.gravity]))
        # The naive implementation assumes that
        l = self.controller_dynamics_params["geom"]["body"]["l"]
        c = math.sqrt(2)/2
        k = self.controller_dynamics_params["motor"]["torque_to_thrust"]
        m = self.controller_dynamics.mass
        # print("Mass: ", m)
        if (self.naive):
            # no feedforward and no thrust mixing
            A = np.array([[c*l, -c*l, -c*l, c*l],
                 [-c*l, -c*l, c*l, c*l],
                 [k, -k, k, -k],
                 [1/m, 1/m, 1/m, 1/m]])
            # print("A Matrix: ", A)
            A_inv = np.linalg.inv(A)
            # print("Desired Torque: ", torque_desired)
            # print("Desired Thrust: ", c_desired)

            desired_states = np.append(torque_desired, c_desired)
            # print("Desired State: ", desired_states)
            # print("A inverse: ", A_inv)
            actions = A_inv @ desired_states
            
            # actions = ((m*c_desired) / 4) * np.ones(4)
            
        else:
            #Initialiaze thrusts
            for i in range(4):
                actions[i] = (self.controller_dynamics.mass * c_desired) / 4
        # self.normalize_thrust()
        self.current_thrust = actions
        print("Desired Thrust: ", actions)
        
        return actions
        
    def set_desired_body_rate(self, goal):
        self.w_des[0] = goal[9]
        self.w_des[1] = goal[10]
        self.w_des[2] = goal[11]
        
    def compute_body_torque(self):
        """
        Compute desired body torque.
        """
        l = self.controller_dynamics_params["geom"]["body"]["l"]
        c = math.sqrt(2)/2
        k = self.controller_dynamics_params["motor"]["torque_to_thrust"]
        
        # Use individual motor thrusts to calculate body torque
        A = np.array([[c*l, -c*l, -c*l, c*l],
                    [-c*l, -c*l, c*l, c*l],
                    [k, -k, k, -k]])
        return np.dot(A, self.current_thrust)
        
        
    def compute_desired_torque(self):
        """
        This is basically the LQR controller.
        """
        body_rate = np.array([self.controller_dynamics.omega[0], self.controller_dynamics.omega[1], self.controller_dynamics.omega[2]])
        
        self.set_desired_body_rate(self.current_goal)

        current_body_torque = self.compute_body_torque()
        
        # Compute Desired body torque from dynamics

        n_ref = np.cross(self.w_des, (self.J @ self.w_des.T))
        # print("w_des: ", self.w_des)
        # print("Body Rate: ", body_rate)
        # print("n_ref: ", n_ref)
        # print("Body Torque: ", current_body_torque)
        # print("Rate Error: ", n_ref - current_body_torque)
        error_vector = np.concatenate([(self.w_des - body_rate), (n_ref - current_body_torque)])
        # 
        # This calculation does not use the feedforward.
        # TODO: Add in first order differentiation for feeforward from Goal point.
        # print("Error vector: ", error_vector.T)
        # print("Dot: ", self.J @ body_rate.T)
        feedforward = np.cross(np.transpose(body_rate), (self.J @ body_rate.T))
        n_des = np.dot(self.K_lqr, np.transpose(error_vector))
        # print("Torque Desired:  ", n_des)
        return n_des
    
    def normalize_thrust(self):
        """
        Since we calculate raw thrust in Newtons, we need to normalize to the max thrust of the crazyflie. 
        """
        motor_linearity = self.controller_dynamics_params["motor"]["linearity"]
        self.current_thrust= self.thrust_max * self.angvel2thrust(self.current_thrust, motor_linearity)
        
    def angvel2thrust(self, w, linearity=0.424):
        """
        CrazyFlie: linearity=0.424
        Args:
            w: thrust_cmds_damp
            linearity (float): linearity factor factor [0 .. 1].
        """
        return (1 - linearity) * w ** 2 + linearity * w
    def print_dynamics(self):
        print(self.controller_dynamics.rot)
        print(self.controller_dynamics.omega)
        print(self.controller_dynamics.thrust_vector)
            
            
    

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
        print("Thrust Control Pre: ", action)
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        print("action in control step: ", action)
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
        import tensorflow as tf
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

        self.tf_control = tf_control
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
