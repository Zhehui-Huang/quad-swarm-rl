from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *
from gym_art.quadrotor_multi.control.utils import quadrotor_jacobian

GRAV = 9.81


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
        to_goal = goal - dynamics.pos
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

