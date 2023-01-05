#!/usr/bin/env python
"""
Quadrotor simulation for OpenAI Gym, with components reusable elsewhere.
Also see: D. Mellinger, N. Michael, V.Kumar. 
Trajectory Generation and Control for Precise Aggressive Maneuvers with Quadrotors
http://journals.sagepub.com/doi/pdf/10.1177/0278364911434236

Developers:
James Preiss, Artem Molchanov, Tao Chen 

References:
[1] RotorS: https://www.researchgate.net/profile/Fadri_Furrer/publication/309291237_RotorS_-_A_Modular_Gazebo_MAV_Simulator_Framework/links/5a0169c4a6fdcc82a3183f8f/RotorS-A-Modular-Gazebo-MAV-Simulator-Framework.pdf
[2] CrazyFlie modelling: http://mikehamer.info/assets/papers/Crazyflie%20Modelling.pdf
[3] HummingBird: http://www.asctec.de/en/uav-uas-drones-rpas-roav/asctec-hummingbird/
[4] CrazyFlie thrusters transition functions: https://www.bitcraze.io/2015/02/measuring-propeller-rpm-part-3/
[5] HummingBird modelling: https://digitalrepository.unm.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1189&context=ece_etds
[6] Rotation b/w matrices: http://www.boris-belousov.net/2016/12/01/quat-dist/#using-rotation-matrices
[7] Rodrigues' rotation formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
"""
import argparse
import logging
import sys
import time

import gym_art.quadrotor_multi.get_state as get_state

# MY LIBS
import gym_art.quadrotor_multi.quadrotor_randomization as quad_rand

# MATH
import matplotlib.pyplot as plt
import transforms3d as t3d

# GYM
from gym.utils import seeding
from gym_art.quadrotor_multi.inertia import QuadLink, QuadLinkSimplified
from gym_art.quadrotor_multi.quadrotor_control import *
from gym_art.quadrotor_multi.quadrotor_visualization import *
from gym_art.quadrotor_multi.sensor_noise import SensorNoise
from gym_art.quadrotor_multi.numba_utils import *

# Numba
from numba import njit

logger = logging.getLogger(__name__)

GRAV = 9.81  # default gravitational constant
EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


# WARN:
# - linearity is set to 1 always, by means of check_quad_param_limits().
# The def. value of linarity for CF is set to 1 as well (due to firmware nonlinearity compensation)

class QuadrotorDynamics:
    """
    Simple simulation of quadrotor dynamics.
    mass unit: kilogram
    arm_length unit: meter
    inertia unit: kg * m^2, 3-element vector representing diagonal matrix
    thrust_to_weight is the total, it will be divided among the 4 props
    torque_to_thrust is ratio of torque produced by prop to thrust
    thrust_noise_ratio is noise2signal ratio of the thrust noise, Ex: 0.05 = 5% of the current signal
      It is an approximate ratio, i.e. the upper bound could still be higher, due to how OU noise operates
    Coord frames: x configuration:
     - x axis between arms looking forward [x - configuration]
     - y axis pointing to the left
     - z axis up
    TODO:
    - only diagonal inertia is used at the moment
    """

    def __init__(self, model_params,
                 room_box=None,
                 dynamics_steps_num=1,
                 dim_mode="3D",
                 gravity=GRAV,
                 dynamics_simplification=False,
                 use_numba=False):

        self.dynamics_steps_num = dynamics_steps_num
        self.dynamics_simplification = dynamics_simplification
        self.use_numba = use_numba
        ###############################################################
        ## PARAMETERS
        self.prop_ccw = np.array([-1., 1., -1., 1.])
        # cw = 1 ; ccw = -1 [ccw, cw, ccw, cw]
        # Reference: https://docs.google.com/document/d/1wZMZQ6jilDbj0JtfeYt0TonjxoMPIgHwYbrFrMNls84/edit
        self.omega_max = 40.  # rad/s The CF sensor can only show 35 rad/s (2000 deg/s), we allow some extra
        self.vxyz_max = 3.  # m/s
        self.gravity = gravity
        self.acc_max = 3. * GRAV

        ###############################################################
        ## Internal State variables
        self.since_last_ort_check = 0  # counter
        self.since_last_ort_check_limit = 0.04  # when to check for non-orthogonality

        self.rot_nonort_limit = 0.01  # How much of non-orthogonality in the R matrix to tolerate
        self.rot_nonort_coeff_maxsofar = 0.  # Statistics on the max number of nonorthogonality that we had

        self.since_last_svd = 0  # counter
        self.since_last_svd_limit = 0.5  # in sec - how ofthen mandatory orthogonalization should be applied

        self.eye = np.eye(3)
        ###############################################################
        ## Initializing model
        self.thrust_noise = None
        self.update_model(model_params)

        ## Sanity checks
        assert self.inertia.shape == (3,)

        ###############################################################
        ## OTHER PARAMETERS
        if room_box is None:
            self.room_box = np.array([[0., 0., 0.], [10., 10., 10.]])
        else:
            self.room_box = np.array(room_box).copy()

        ## Selecting 1D, Planar or Full 3D modes
        self.dim_mode = dim_mode
        if self.dim_mode == '1D':
            self.control_mx = np.ones([4, 1])
        elif self.dim_mode == '2D':
            self.control_mx = np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        elif self.dim_mode == '3D':
            self.control_mx = np.eye(4)
        else:
            raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)

    @staticmethod
    def angvel2thrust(w, linearity=0.424):
        """
        Args:
            linearity (float): linearity factor factor [0 .. 1].
            CrazyFlie: linearity=0.424
        """
        return (1 - linearity) * w ** 2 + linearity * w

    def update_model(self, model_params):
        if self.dynamics_simplification:
            self.model = QuadLinkSimplified(params=model_params["geom"])
        else:
            self.model = QuadLink(params=model_params["geom"])
        self.model_params = model_params

        ###############################################################
        ## PARAMETERS FOR RANDOMIZATION
        self.mass = self.model.m
        self.inertia = np.diagonal(self.model.I_com)

        self.thrust_to_weight = self.model_params["motor"]["thrust_to_weight"]
        self.torque_to_thrust = self.model_params["motor"]["torque_to_thrust"]
        self.motor_linearity = self.model_params["motor"]["linearity"]
        self.C_rot_drag = self.model_params["motor"]["C_drag"]
        self.C_rot_roll = self.model_params["motor"]["C_roll"]
        self.motor_damp_time_up = self.model_params["motor"]["damp_time_up"]
        self.motor_damp_time_down = self.model_params["motor"]["damp_time_down"]

        self.thrust_noise_ratio = self.model_params["noise"]["thrust_noise_ratio"]
        self.vel_damp = self.model_params["damp"]["vel"]
        self.damp_omega_quadratic = self.model_params["damp"]["omega_quadratic"]

        ###############################################################
        ## COMPUTED (Dependent) PARAMETERS
        try:
            self.motor_assymetry = np.array(self.model_params["motor"]["assymetry"])
        except:
            self.motor_assymetry = np.array([1.0, 1.0, 1.0, 1.0])
            print("WARNING: Motor assymetry was not setup. Setting assymetry to:", self.motor_assymetry)
        self.motor_assymetry = self.motor_assymetry * 4. / np.sum(self.motor_assymetry)  # re-normalizing to sum-up to 4
        self.thrust_max = GRAV * self.mass * self.thrust_to_weight * self.motor_assymetry / 4.0
        self.torque_max = self.torque_to_thrust * self.thrust_max  # propeller torque scales

        # Propeller positions in X configurations
        self.prop_pos = self.model.prop_pos

        # unit: meters^2 ??? maybe wrong
        self.prop_crossproducts = np.cross(self.prop_pos, [0., 0., 1.])
        self.prop_ccw_mx = np.zeros([3, 4])  # Matrix allows using matrix multiplication
        self.prop_ccw_mx[2, :] = self.prop_ccw

        ## Forced dynamics auxiliary matrices
        # Prop crossproduct give torque directions
        self.G_omega_thrust = self.thrust_max * self.prop_crossproducts.T  # [3,4] @ [4,1]
        # additional torques along z-axis caused by propeller rotations
        self.C_omega_prop = self.torque_max * self.prop_ccw_mx  # [3,4] @ [4,1] = [3,1]
        self.G_omega = (1.0 / self.inertia)[:, None] * (self.G_omega_thrust + self.C_omega_prop)

        # Allows to sum-up thrusts as a linear matrix operation
        self.thrust_sum_mx = np.zeros([3, 4])  # [0,0,F_sum].T
        self.thrust_sum_mx[2, :] = 1  # [0,0,F_sum].T

        self.init_thrust_noise()

        self.arm = np.linalg.norm(self.model.motor_xyz[:2])

        ## the ratio between max torque and inertia around each axis
        ## the 0-1 matrix on the right is the way to sum-up
        self.torque_to_inertia = self.G_omega @ np.array([[0, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]])
        self.torque_to_inertia = np.sum(self.torque_to_inertia, axis=1)
        # self.torque_to_inertia = self.torque_to_inertia / np.linalg.norm(self.torque_to_inertia)

        self.reset()

    def init_thrust_noise(self):
        # sigma = 0.2 gives roughly max noise of -1 .. 1
        if self.use_numba:
            self.thrust_noise = OUNoiseNumba(4, sigma=0.2 * self.thrust_noise_ratio)
        else:
            self.thrust_noise = OUNoise(4, sigma=0.2 * self.thrust_noise_ratio)

    # pos, vel, in world coords (meters)
    # rotation is 3x3 matrix (body coords) -> (world coords)dt
    # omega is angular velocity (radians/sec) in body coords, i.e. the gyroscope
    def set_state(self, position, velocity, rotation, omega, thrusts=np.zeros((4,))):
        for v in (position, velocity, omega):
            assert v.shape == (3,)
        assert thrusts.shape == (4,)
        assert rotation.shape == (3, 3)
        self.pos = deepcopy(position)
        self.vel = deepcopy(velocity)
        self.acc = np.zeros(3)
        self.accelerometer = np.array([0, 0, GRAV])
        self.rot = deepcopy(rotation)
        self.omega = deepcopy(omega.astype(np.float32))
        self.thrusts = deepcopy(thrusts)

    # generate a random state (meters, meters/sec, radians/sec)
    def random_state(self, box, vel_max=15.0, omega_max=2 * np.pi):
        box = np.array(box)
        pos = np.random.uniform(low=-box, high=box, size=(3,))

        vel = np.random.uniform(low=-vel_max, high=vel_max, size=(3,))
        vel_magn = np.random.uniform(low=0., high=vel_max)
        vel = vel_magn / (np.linalg.norm(vel) + EPS) * vel

        omega = np.random.uniform(low=-omega_max, high=omega_max, size=(3,))
        omega_magn = np.random.uniform(low=0., high=omega_max)
        omega = omega_magn / (np.linalg.norm(omega) + EPS) * omega

        rot = rand_uniform_rot3d()
        return pos, vel, rot, omega
        # self.set_state(pos, vel, rot, omega)

    # generate a random state (meters, meters/sec, radians/sec)
    def pitch_roll_restricted_random_state(self, box, vel_max=15.0, omega_max=2 * np.pi, pitch_max=0.5, roll_max=0.5,
                                           yaw_max=3.14):
        pos = np.random.uniform(low=-box, high=box, size=(3,))

        vel = np.random.uniform(low=-vel_max, high=vel_max, size=(3,))
        vel_magn = np.random.uniform(low=0., high=vel_max)
        vel = vel_magn / (np.linalg.norm(vel) + EPS) * vel

        omega = np.random.uniform(low=-omega_max, high=omega_max, size=(3,))
        omega_magn = np.random.uniform(low=0., high=omega_max)
        omega = omega_magn / (np.linalg.norm(omega) + EPS) * omega

        pitch = np.random.uniform(low=-pitch_max, high=pitch_max)
        roll = np.random.uniform(low=-roll_max, high=roll_max)
        yaw = np.random.uniform(low=-yaw_max, high=yaw_max)
        rot = t3d.euler.euler2mat(roll, pitch, yaw)
        return pos, vel, rot, omega

    def step(self, thrust_cmds, dt):
        thrust_noise = self.thrust_noise.noise()

        if self.use_numba:
            [self.step1_numba(thrust_cmds, dt, thrust_noise) for t in range(self.dynamics_steps_num)]
        else:
            [self.step1(thrust_cmds, dt, thrust_noise) for t in range(self.dynamics_steps_num)]

    ## Step function integrates based on current derivative values (best fits affine dynamics model)
    # thrust_cmds is motor thrusts given in normalized range [0, 1].
    # 1 represents the max possible thrust of the motor.
    ## Frames:
    # pos - global
    # vel - global
    # rot - global
    # omega - body frame
    # goal_pos - global
    def step1(self, thrust_cmds, dt, thrust_noise):
        # print("thrust_cmds:", thrust_cmds)
        # uncomment for debugging. they are slow
        # assert np.all(thrust_cmds >= 0)
        # assert np.all(thrust_cmds <= 1)

        thrust_cmds = np.clip(thrust_cmds, a_min=0., a_max=1.)
        ###################################
        ## Filtering the thruster and adding noise
        # I use the multiplier 4, since 4*T ~ time for a step response to finish, where
        # T is a time constant of the first-order filter
        self.motor_tau_up = 4 * dt / (self.motor_damp_time_up + EPS)
        self.motor_tau_down = 4 * dt / (self.motor_damp_time_down + EPS)
        motor_tau = self.motor_tau_up * np.ones([4, ])
        motor_tau[thrust_cmds < self.thrust_cmds_damp] = self.motor_tau_down
        motor_tau[motor_tau > 1.] = 1.

        ## Since NN commands thrusts we need to convert to rot vel and back
        # WARNING: Unfortunately if the linearity != 1 then filtering using square root is not quite correct
        # since it likely means that you are using rotational velocities as an input instead of the thrust and hence
        # you are filtering square roots of angular velocities
        thrust_rot = thrust_cmds ** 0.5
        self.thrust_rot_damp = motor_tau * (thrust_rot - self.thrust_rot_damp) + self.thrust_rot_damp
        self.thrust_cmds_damp = self.thrust_rot_damp ** 2

        ## Adding noise
        thrust_noise = thrust_cmds * thrust_noise
        self.thrust_cmds_damp = np.clip(self.thrust_cmds_damp + thrust_noise, 0.0, 1.0)

        thrusts = self.thrust_max * self.angvel2thrust(self.thrust_cmds_damp, linearity=self.motor_linearity)
        # Prop crossproduct give torque directions
        self.torques = self.prop_crossproducts * thrusts[:, None]  # (4,3)=(props, xyz)
        
        # additional torques along z-axis caused by propeller rotations
        self.torques[:, 2] += self.torque_max * self.prop_ccw * self.thrust_cmds_damp

        # net torque: sum over propellers
        thrust_torque = np.sum(self.torques, axis=0)

        ###################################
        ## Rotor drag and Rolling forces and moments
        ## See Ref[1] Sec:2.1 for detailes

        # self.C_rot_drag = 0.0028
        # self.C_rot_roll = 0.003 # 0.0003
        if self.C_rot_drag != 0 or self.C_rot_roll != 0:
            # self.vel = np.zeros_like(self.vel)
            # v_rotors[3,4]  = (rot[3,3] @ vel[3,])[3,] + (omega[3,] x prop_pos[4,3])[4,3]
            # v_rotors = self.rot.T @ self.vel + np.cross(self.omega, self.model.prop_pos)
            vel_body = self.rot.T @ self.vel
            v_rotors = vel_body + cross_vec_mx4(self.omega, self.model.prop_pos)
            # assert v_rotors.shape == (4,3)
            v_rotors[:, 2] = 0.  # Projection to the rotor plane

            # Drag/Roll of rotors (both in body frame)
            rotor_drag_fi = - self.C_rot_drag * np.sqrt(self.thrust_cmds_damp)[:, None] * v_rotors  # [4,3]
            rotor_drag_force = np.sum(rotor_drag_fi, axis=0)
            # rotor_drag_ti = np.cross(rotor_drag_fi, self.model.prop_pos)#[4,3] x [4,3]
            rotor_drag_ti = cross_mx4(rotor_drag_fi, self.model.prop_pos)  # [4,3] x [4,3]
            rotor_drag_torque = np.sum(rotor_drag_ti, axis=0)

            rotor_roll_torque = - self.C_rot_roll * self.prop_ccw[:, None] * np.sqrt(self.thrust_cmds_damp)[:,
                                                                             None] * v_rotors  # [4,3]
            rotor_roll_torque = np.sum(rotor_roll_torque, axis=0)
            rotor_visc_torque = rotor_drag_torque + rotor_roll_torque

            ## Constraints (prevent numerical instabilities)
            vel_norm = np.linalg.norm(vel_body)
            rdf_norm = np.linalg.norm(rotor_drag_force)
            rdf_norm_clip = np.clip(rdf_norm, a_min=0., a_max=vel_norm * self.mass / (2 * dt))
            if rdf_norm > EPS:
                rotor_drag_force = (rotor_drag_force / rdf_norm) * rdf_norm_clip

            # omega_norm = np.linalg.norm(self.omega)
            rvt_norm = np.linalg.norm(rotor_visc_torque)
            rvt_norm_clipped = np.clip(rvt_norm, a_min=0., a_max=np.linalg.norm(self.omega * self.inertia) / (2 * dt))
            if rvt_norm > EPS:
                rotor_visc_torque = (rotor_visc_torque / rvt_norm) * rvt_norm_clipped

            # print("v", self.vel, "\nomega:\n",self.omega, "\nv_rotors:\n", v_rotors, "\nrotor_drag_fi:\n", rotor_drag_fi)
            # if rvt_norm_clipped/rvt_norm < 1 or rdf_norm_clip/rdf_norm < 1:
            #     print("Clip:", rvt_norm_clipped/rvt_norm, rdf_norm_clip/rdf_norm)
            #     print("---------------------------------------------------------------")
        else:
            rotor_visc_torque = rotor_drag_torque = rotor_drag_force = rotor_roll_torque = np.zeros(3)

        ###################################
        ## (Square) Damping using torques (in case we would like to add damping using torques)
        # damping_torque = - 0.3 * self.omega * np.fabs(self.omega)
        self.torque = thrust_torque + rotor_visc_torque
        thrust = npa(0, 0, np.sum(thrusts))

        #########################################################
        ## ROTATIONAL DYNAMICS

        ###################################
        ## Integrating rotations (based on current values)
        omega_vec = np.matmul(self.rot, self.omega)  # Change from body2world frame
        wx, wy, wz = omega_vec
        omega_norm = np.linalg.norm(omega_vec)
        if omega_norm != 0:
            # See [7]
            K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
            rot_angle = omega_norm * dt
            dRdt = self.eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
            self.rot = dRdt @ self.rot

        ## SVD is not strictly required anymore. Performing it rarely, just in case
        self.since_last_svd += dt
        if self.since_last_svd > self.since_last_svd_limit:
            ## Perform SVD orthogonolization
            u, s, v = np.linalg.svd(self.rot)
            self.rot = np.matmul(u, v)
            self.since_last_svd = 0

        ###################################
        ## COMPUTING OMEGA UPDATE

        ## Damping using velocities (I find it more stable numerically)
        ## Linear damping

        # This is only for linear damping of angular velocity.
        # omega_damp = 0.999
        # self.omega = omega_damp * self.omega + dt * omega_dot

        self.omega_dot = ((1.0 / self.inertia) *
                          (cross(-self.omega, self.inertia * self.omega) + self.torque))

        ## Quadratic damping
        # 0.03 corresponds to roughly 1 revolution per sec
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * self.omega ** 2, a_min=0.0, a_max=1.0)
        self.omega = self.omega + (1.0 - omega_damp_quadratic) * dt * self.omega_dot
        self.omega = np.clip(self.omega, a_min=-self.omega_max, a_max=self.omega_max)
        
        if self.pos[2] <= 0.05:
            self.omega = np.zeros(3)
            self.rot = self.eye

        ## When use square damping on torques - use simple integration
        ## since damping is accounted as part of the net torque
        # self.omega += dt * omega_dot

        #########################################################
        # TRANSLATIONAL DYNAMICS

        ## Room constraints
        mask = np.logical_or(self.pos <= self.room_box[0], self.pos >= self.room_box[1])

        ## Computing position
        self.pos = self.pos + dt * self.vel

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        self.pos_before_clip = self.pos.copy()
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])
        # self.vel[np.equal(self.pos, self.pos_before_clip)] = 0.

        ## Computing accelerations
        acc = [0, 0, -GRAV] + (1.0 / self.mass) * np.matmul(self.rot, (thrust + rotor_drag_force))
        # acc[mask] = 0. #If we leave the room - stop accelerating

        # Add friction if the drones is on the ground
        # friction = mu_k * F_n
        static_mu_k = 99999.
        rolling_mu_k = 99999.
        if self.pos[2] <= 0.05:
            # step 0: force pos_z = 0.05
            self.pos[2] = 0.05
            # step 1: compute F_n
            vel_xy = np.array([self.vel[0], self.vel[1], 0.])
            vel_xy_dir = vel_xy / np.linalg.norm(vel_xy)
            acc_xy = np.array([acc[0], acc[1], 0.])
            acc_xy_dir = acc_xy / np.linalg.norm(acc_xy)
            F_n = -(-GRAV + acc[2]) * self.mass

            # step 2: determine static or rolling friction
            F_xy = acc_xy * self.mass
            if np.linalg.norm(vel_xy) == 0:
                # step 2.1: static friction
                # check if exceeds limit static friction
                max_static_friction = - acc_xy_dir * F_n * static_mu_k
                if np.dot(F_xy + max_static_friction, F_xy) > 0:
                    acc += max_static_friction / self.mass
                else:
                    acc[0:2] = 0.
            else:
                # step 2.2: rolling friction
                # check if stops with dt
                rolling_friction = - vel_xy_dir * F_n * rolling_mu_k
                acc += rolling_friction / self.mass
                acc_xy = np.array([acc[0], acc[1], 0.])
                pred_vel_xy = vel_xy + dt * acc_xy
                if np.dot(vel_xy, pred_vel_xy) <= 0:
                    acc[0:2] = 0.
                    self.vel[0:2] = 0.
        self.acc = acc

        ## Computing velocities
        self.vel = (1.0 - self.vel_damp) * self.vel + dt * acc
        # self.vel[mask] = 0. #If we leave the room - stop flying

        ## Accelerometer measures so called "proper acceleration"
        # that includes gravity with the opposite sign
        self.accelerometer = np.matmul(self.rot.T, acc + [0, 0, self.gravity])

    def step1_numba(self, thrust_cmds, dt, thrust_noise):
        self.motor_tau_up, self.motor_tau_down, self.thrust_rot_damp, self.thrust_cmds_damp, self.torques, \
        self.torque, self.rot, self.since_last_svd, self.omega_dot, self.omega, self.pos, thrust, rotor_drag_force = \
            calculate_torque_integrate_rotations_and_update_omega(thrust_cmds, dt, EPS, self.motor_damp_time_up, self.motor_damp_time_down,
                         self.thrust_cmds_damp, self.thrust_rot_damp, thrust_noise, self.thrust_max, self.motor_linearity,
                         self.prop_crossproducts, self.prop_ccw, self.torque_max, self.rot, np.float64(self.omega),
                         self.eye, self.since_last_svd, self.since_last_svd_limit, self.inertia,
                         self.damp_omega_quadratic, self.omega_max, self.pos, self.vel)

        # Clipping if met the obstacle and nullify velocities (not sure what to do about accelerations)
        self.pos = np.clip(self.pos, a_min=self.room_box[0], a_max=self.room_box[1])

        # Set constant variables up for numba
        grav_cnst_arr = np.float64([0, 0, -GRAV])
        sum_thr_drag = thrust + rotor_drag_force
        grav_arr = np.float64([0, 0, self.gravity])
        self.vel, self.acc, self.accelerometer = compute_velocity_and_acceleration(self.vel, grav_cnst_arr, self.mass, self.rot,
                                                                         sum_thr_drag, self.vel_damp, dt, self.rot.T,
                                                                         grav_arr)

    def reset(self):
        self.thrust_cmds_damp = np.zeros([4])
        self.thrust_rot_damp = np.zeros([4])

    def rotors_drag_roll_glob_frame(self):
        # omega [3,] x prop_pos [4,3] = v_rot_body [4, 3]
        # R[3,3] @ prop_pos.T[3,4] = v_rotors[3,4]
        v_rotors = self.vel + self.rot @ np.cross(self.omega, self.model.prop_pos).T
        rot_z = self.rot[:, 2]

        # [3,4] = [3,4] - ([3,4].T @ [3,1]).T * [3,4]
        v_rotors_perp = v_rotors - (v_rotors.T @ rot_z).T * np.repeat(rot_z, 4, axis=1)

        # Drag/Roll of rotors
        rotor_drag = - self.C_rot_drag * np.sqrt(self.thrust_cmds_damp) * v_rotors_perp  # [3,4]
        rotor_roll_torque = self.C_rot_roll * np.sqrt(self.thrust_cmds_damp) * v_rotors_perp  # [3,4]

    #######################################################
    ## AFFINE DYNAMICS REPRESENTATION:
    # s = dt*(F(s) + G(s)*u)

    ## Unforced dynamics (integrator, damping_deceleration)
    def F(self, s, dt):
        xyz = s[0:3]
        Vxyz = s[3:6]
        rot = s[6:15].reshape([3, 3])
        omega = s[15:18]
        goal = s[18:21]

        ###############################
        ## Linear position change
        dx = deepcopy(Vxyz)

        ###############################
        ## Linear velocity change
        dV = ((1.0 - self.vel_damp) * Vxyz - Vxyz) / dt + np.array([0, 0, -GRAV])

        ###############################
        ## Angular orientation change
        omega_vec = np.matmul(rot, omega)  # Change from body2world frame
        wx, wy, wz = omega_vec
        omega_mat_deriv = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])

        # ROtation matrix derivative
        dR = np.matmul(omega_mat_deriv, rot).flatten()

        ###############################
        ## Angular rate change
        F_omega = (1.0 / self.inertia) * (cross(-omega, self.inertia * omega))
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * omega ** 2, a_min=0.0, a_max=1.0)
        dOmega = (1.0 - omega_damp_quadratic) * F_omega

        ###############################
        ## Goal change
        dgoal = np.zeros_like(goal)

        return np.concatenate([dx, dV, dR, dOmega, dgoal])

    ## Forced affine dynamics (controlling acceleration only)
    def G(self, s):
        xyz = s[0:3]
        Vxyz = s[3:6]
        rot = s[6:15].reshape([3, 3])
        omega = s[15:18]
        goal = s[18:21]

        ###############################
        ## dx, dV, dR, dgoal
        dx = np.zeros([3, 4])
        dV = (rot / self.mass) @ (self.thrust_max * self.thrust_sum_mx)
        dR = np.zeros([9, 4])
        dgoal = np.zeros([3, 4])

        ###############################
        ## Angular acceleration
        omega_damp_quadratic = np.clip(self.damp_omega_quadratic * omega ** 2, a_min=0.0, a_max=1.0)
        dOmega = (1.0 - omega_damp_quadratic)[:, None] * self.G_omega

        return np.concatenate([dx, dV, dR, dOmega, dgoal], axis=0) @ self.control_mx

    # return eye, center, up suitable for gluLookAt representing onboard camera
    def look_at(self):
        degrees_down = 45.0
        R = self.rot
        # camera slightly below COM
        eye = self.pos + np.matmul(R, [0, 0, -0.02])
        theta = np.radians(degrees_down)
        to, _ = normalize(np.cos(theta) * R[:, 0] - np.sin(theta) * R[:, 2])
        center = eye + to
        up = cross(to, R[:, 1])
        return eye, center, up

    def state_vector(self):
        return np.concatenate([
            self.pos, self.vel, self.rot.flatten(), self.omega])

    @staticmethod
    def action_space():
        low = np.zeros(4)
        high = np.ones(4)
        return spaces.Box(low, high, dtype=np.float32)

    def __deepcopy__(self, memo):
        """Certain numba-optimized instance attributes can't be naively copied."""

        cls = self.__class__
        copied_dynamics = cls.__new__(cls)
        memo[id(self)] = copied_dynamics

        skip_copying = {"thrust_noise"}

        for k, v in self.__dict__.items():
            if k not in skip_copying:
                setattr(copied_dynamics, k, deepcopy(v, memo))

        copied_dynamics.init_thrust_noise()
        return copied_dynamics


# reasonable reward function for hovering at a goal and not flying too high
def compute_reward_weighted(dynamics, goal, action, dt, crashed, time_remain, rew_coeff, action_prev,
                            quads_settle=False, quads_settle_range_meters=1.0, quads_vel_reward_out_range=0.8):
    ##################################################
    ## log to create a sharp peak at the goal
    dist = np.linalg.norm(goal - dynamics.pos)
    cost_pos_raw = dist
    cost_pos = rew_coeff["pos"] * cost_pos_raw

    # sphere of equal reward if drones are close to the goal position
    vel_coeff = rew_coeff["vel"]
    if dist <= quads_settle_range_meters and quads_settle:
        cost_pos = 0
        vel_coeff = quads_vel_reward_out_range  # penalize movement once drones are close to the goal

    ##################################################
    # penalize amount of control effort
    cost_effort_raw = np.linalg.norm(action)
    cost_effort = rew_coeff["effort"] * cost_effort_raw

    dact = action - action_prev
    cost_act_change_raw = (dact[0] ** 2 + dact[1] ** 2 + dact[2] ** 2 + dact[3] ** 2) ** 0.5
    cost_act_change = rew_coeff["action_change"] * cost_act_change_raw

    ##################################################
    ## loss velocity
    cost_vel_raw = np.linalg.norm(dynamics.vel)
    cost_vel = vel_coeff * cost_vel_raw

    ##################################################
    ## Loss orientation
    cost_orient_raw = -dynamics.rot[2, 2]
    cost_orient = rew_coeff["orient"] * cost_orient_raw

    cost_yaw_raw = -dynamics.rot[0, 0]
    cost_yaw = rew_coeff["yaw"] * cost_yaw_raw

    # Projection of the z-body axis to z-world axis
    # Negative, because the larger the projection the smaller the loss (i.e. the higher the reward)
    rot_cos = ((dynamics.rot[0, 0] + dynamics.rot[1, 1] + dynamics.rot[2, 2]) - 1.) / 2.
    # We have to clip since rotation matrix falls out of orthogonalization from time to time
    cost_rotation_raw = np.arccos(np.clip(rot_cos, -1., 1.))  # angle = arccos((trR-1)/2) See: [6]
    cost_rotation = rew_coeff["rot"] * cost_rotation_raw

    cost_attitude_raw = np.arccos(np.clip(dynamics.rot[2, 2], -1., 1.))
    cost_attitude = rew_coeff["attitude"] * cost_attitude_raw

    ##################################################
    ## Loss for constant uncontrolled rotation around vertical axis
    cost_spin_raw = (dynamics.omega[0] ** 2 + dynamics.omega[1] ** 2 + dynamics.omega[2] ** 2) ** 0.5
    cost_spin = rew_coeff["spin"] * cost_spin_raw

    ##################################################
    # loss crash
    cost_crash_raw = float(crashed)
    cost_crash = rew_coeff["crash"] * cost_crash_raw

    reward = -dt * np.sum([
        cost_pos,
        cost_effort,
        cost_crash,
        cost_orient,
        cost_yaw,
        cost_rotation,
        cost_attitude,
        cost_spin,
        cost_act_change,
        cost_vel
    ])

    rew_info = {
        "rew_main": -cost_pos,
        'rew_pos': -cost_pos,
        'rew_action': -cost_effort,
        'rew_crash': -cost_crash,
        "rew_orient": -cost_orient,
        "rew_yaw": -cost_yaw,
        "rew_rot": -cost_rotation,
        "rew_attitude": -cost_attitude,
        "rew_spin": -cost_spin,
        "rew_act_change": -cost_act_change,
        "rew_vel": -cost_vel,

        "rewraw_main": -cost_pos_raw,
        'rewraw_pos': -cost_pos_raw,
        'rewraw_action': -cost_effort_raw,
        'rewraw_crash': -cost_crash_raw,
        "rewraw_orient": -cost_orient_raw,
        "rewraw_yaw": -cost_yaw_raw,
        "rewraw_rot": -cost_rotation_raw,
        "rewraw_attitude": -cost_attitude_raw,
        "rewraw_spin": -cost_spin_raw,
        "rewraw_act_change": -cost_act_change_raw,
        "rewraw_vel": -cost_vel_raw,
    }

    # report rewards in the same format as they are added to the actual agent's reward (easier to debug this way)
    for k, v in rew_info.items():
        rew_info[k] = dt * v

    if np.isnan(reward) or not np.isfinite(reward):
        for key, value in locals().items():
            print('%s: %s \n' % (key, str(value)))
        raise ValueError('QuadEnv: reward is Nan')

    return reward, rew_info


####################################################################################################################################################################
## ENV
# Gym environment for quadrotor seeking the origin with no obstacles and full state observations.
# NOTES:
# - room size of the env and init state distribution are not the same !
#   It is done for the reason of having static (and preferably short) episode length, since for some distance it would be impossible to reach the goal
class QuadrotorSingle:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, dynamics_params="DefaultQuad", dynamics_change=None,
                 dynamics_randomize_every=None, dyn_sampler_1=None, dyn_sampler_2=None,
                 raw_control=True, raw_control_zero_middle=True, dim_mode='3D', tf_control=False, sim_freq=200.,
                 sim_steps=2,
                 obs_repr="xyz_vxyz_R_omega", ep_time=7, obstacles_num=0, room_length=10, room_width=10, room_height=10, init_random_state=False,
                 rew_coeff=None, sense_noise=None, verbose=False, gravity=GRAV,
                 t2w_std=0.005, t2t_std=0.0005, excite=False, dynamics_simplification=False, use_numba=False, swarm_obs='none', num_agents=1,quads_settle=False,
                 quads_settle_range_meters=1.0, quads_vel_reward_out_range=0.8,
                 view_mode='local', obstacle_mode='no_obstacles', obstacle_num=0, num_use_neighbor_obs=0):
        np.seterr(under='ignore')
        """
        Args:
            dynamics_params: [str or dict] loading dynamics params by name or by providing a dictionary. 
                If "random": dynamics will be randomized completely (see sample_dyn_parameters() )
                If dynamics_randomize_every is None: it will be randomized only once at the beginning.
                One can randomize dynamics during the end of any episode using resample_dynamics()
                WARNING: randomization during an episode is not supported yet. Randomize ONLY before calling reset().
            dynamics_change: [dict] update to dynamics parameters relative to dynamics_params provided
            
            dynamics_randomize_every: [int] how often (trajectories) perform randomization
            dynamics_sampler_1: [dict] the first sampler to be applied. Dict must contain type (see quadrotor_randomization) and whatever params samler requires
            dynamics_sampler_2: [dict] the second sampler to be applied. Convenient if you need to fix some params after sampling.
            
            raw_control: [bool] use raw cantrol or the Mellinger controller as a default
            raw_control_zero_middle: [bool] meaning that control will be [-1 .. 1] rather than [0 .. 1]
            dim_mode: [str] Dimensionality of the env. Options: 1D(just a vertical stabilization), 2D(vertical plane), 3D(normal)
            tf_control: [bool] creates Mellinger controller using TensorFlow
            sim_freq (float): frequency of simulation
            sim_steps: [int] how many simulation steps for each control step
            obs_repr: [str] options: xyz_vxyz_rot_omega, xyz_vxyz_quat_omega
            ep_time: [float] episode time in simulated seconds. This parameter is used to compute env max time length in steps.
            obstacles_num: [int] number of obstacle in the env
            room_size: [int] env room size. Not the same as the initialization box to allow shorter episodes
            init_random_state: [bool] use random state initialization or horizontal initialization with 0 velocities
            rew_coeff: [dict] weights for different reward components (see compute_weighted_reward() function)
            sens_noise (dict or str): sensor noise parameters. If None - no noise. If "default" then the default params are loaded. Otherwise one can provide specific params.
            excite: [bool] change the setpoint at the fixed frequency to perturb the quad
        """
        ## ARGS
        self.init_random_state = init_random_state
        self.room_length = room_length
        self.room_width = room_width
        self.room_height = room_height
        self.room_size = room_length * room_width * room_height
        self.obs_repr = obs_repr
        self.sim_steps = sim_steps
        self.dim_mode = dim_mode
        self.raw_control_zero_middle = raw_control_zero_middle
        self.tf_control = tf_control
        self.dynamics_randomize_every = dynamics_randomize_every
        self.verbose = verbose
        self.obstacles_num = obstacles_num
        self.raw_control = raw_control
        self.use_numba = use_numba
        self.update_sense_noise(sense_noise=sense_noise)
        self.gravity = gravity
        self.swarm_obs = swarm_obs
        self.num_use_neighbor_obs = num_use_neighbor_obs
        self.num_agents = num_agents
        self.quads_settle = quads_settle
        self.quads_settle_range_meters = quads_settle_range_meters
        self.quads_vel_reward_out_range = quads_vel_reward_out_range
        ## t2w and t2t ranges
        self.t2w_std = t2w_std
        self.t2w_min = 1.5
        self.t2w_max = 10.0

        self.t2t_std = t2t_std
        self.t2t_min = 0.005
        self.t2t_max = 1.0
        self.excite = excite
        ## dynmaics simplification
        self.dynamics_simplification = dynamics_simplification
        ## PARAMS
        self.max_init_vel = 1.  # m/s
        self.max_init_omega = 2 * np.pi  # rad/s
        # self.pitch_max = 1. #rad
        # self.roll_max = 1.  #rad
        # self.yaw_max = np.pi   #rad

        self.room_box = np.array(
            [[-self.room_length/2, -self.room_width/2, 0], [self.room_length/2, self.room_width/2, self.room_height]]) # diagonal coordinates of box (?)
        self.state_vector = self.state_vector = getattr(get_state, "state_" + self.obs_repr)

        ## WARN: If you
        # size of the box from which initial position will be randomly sampled
        # if box_scale > 1.0 then it will also growevery episode
        self.box = 2.0
        self.box_scale = 1.0  # scale the initialbox by this factor eache episode

        self.goal = None

        ## Statistics vars
        self.traj_count = 0

        ## View / Camera mode
        self.view_mode = view_mode

        ## Obstacle Mode
        self.obstacle_mode = obstacle_mode
        self.obstacle_num = obstacle_num

        ###############################################################################
        ## DYNAMICS (and randomization)

        # Could be dynamics of a specific quad or a random dynamics (i.e. randomquad)
        self.dyn_base_sampler = getattr(quad_rand, dynamics_params)()
        self.dynamics_change = copy.deepcopy(dynamics_change)

        self.dynamics_params = self.dyn_base_sampler.sample()
        ## Now, updating if we are providing modifications
        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        self.dyn_sampler_1 = dyn_sampler_1
        if dyn_sampler_1 is not None:
            sampler_type = dyn_sampler_1["class"]
            self.dyn_sampler_1_params = copy.deepcopy(dyn_sampler_1)
            del self.dyn_sampler_1_params["class"]
            self.dyn_sampler_1 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_1_params)

        self.dyn_sampler_2 = dyn_sampler_2
        if dyn_sampler_2 is not None:
            sampler_type = dyn_sampler_2["class"]
            self.dyn_sampler_2_params = copy.deepcopy(dyn_sampler_2)
            del self.dyn_sampler_2_params["class"]
            self.dyn_sampler_2 = getattr(quad_rand, sampler_type)(params=self.dynamics_params,
                                                                  **self.dyn_sampler_2_params)

        ## Updating dynamics
        dyn_upd_start_time = time.time()
        ## Also performs update of the dynamics
        self.action_space = None  # to be defined in update_dynamics
        self.resample_dynamics()
        # self.update_dynamics(dynamics_params=self.dynamics_params)
        print("QuadEnv: Dyn update time: ", time.time() - dyn_upd_start_time)

        if self.verbose:
            print("###############################################")
            print("DYN RANDOMIZATION PARAMS:")
            print_dic(self.dyn_randomization_params)
            print("###############################################")
            self.dynamics_params = self.dynamics_params_def

        ###############################################################################
        ## OBSERVATIONS
        self.observation_space = self.make_observation_space()

        ################################################################################
        ## DIMENSIONALITY
        if self.view_mode == 'local':
            if self.dim_mode == '1D' or self.dim_mode == '2D':
                self.viewpoint = 'side'
            else:
                self.viewpoint = 'chase'
        else:
            self.viewpoint = 'global'

        ################################################################################
        ## EPISODE PARAMS
        # TODO get this from a wrapper
        self.ep_time = ep_time  # In seconds
        self.dt = 1.0 / sim_freq
        self.metadata["video.frames_per_second"] = sim_freq / self.sim_steps
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))
        self.tick = 0
        self.crashed = False
        self.control_freq = sim_freq / sim_steps

        self.rew_coeff = None  # provided by the parent multi_env

        #########################################
        self._seed()

    def reset_ep_len(self, ep_time):
        self.ep_time = ep_time
        self.ep_len = int(self.ep_time / (self.dt * self.sim_steps))

    def save_dyn_params(self, filename):
        import yaml
        with open(filename, 'w') as yaml_file:
            def numpy_convert(key, item):
                return str(item)

            self.dynamics_params_converted = copy.deepcopy(self.dynamics_params)
            walk_dict(self.dynamics_params_converted, numpy_convert)
            yaml_file.write(yaml.dump(self.dynamics_params_converted, default_flow_style=False))

    def update_env(self, room_length, room_width, room_height):
        self.room_length, self.room_width, self.room_height = room_length, room_width, room_height
        self.room_box = np.array(
            [[-self.room_length/2, -self.room_width/2, 0], [self.room_length/2, self.room_width/2, self.room_height]])  # diagonal coordinates of box (?)
        self.dynamics.room_box = self.room_box

    def update_sense_noise(self, sense_noise):
        if isinstance(sense_noise, dict):
            self.sense_noise = SensorNoise(**sense_noise)
        elif isinstance(sense_noise, str):
            if sense_noise == "default":
                self.sense_noise = SensorNoise(bypass=False, use_numba=self.use_numba)
            else:
                ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))
        elif sense_noise is None:
            self.sense_noise = SensorNoise(bypass=True)
        else:
            raise ValueError("ERROR: QuadEnv: sense_noise parameter is of unknown type: " + str(sense_noise))

    def update_dynamics(self, dynamics_params):
        ################################################################################
        ## DYNAMICS
        ## Then loading the dynamics
        self.dynamics_params = dynamics_params
        self.dynamics = QuadrotorDynamics(model_params=dynamics_params,
                                          dynamics_steps_num=self.sim_steps, room_box=self.room_box,
                                          dim_mode=self.dim_mode,
                                          gravity=self.gravity, dynamics_simplification=self.dynamics_simplification,
                                          use_numba=self.use_numba)

        if self.verbose:
            print("#################################################")
            print("Dynamics params loaded:")
            print_dic(dynamics_params)
            print("#################################################")

        ################################################################################
        ## SCENE
        if self.obstacles_num > 0:
            #TODO: Fix unresolved reference error here???
            self.obstacles = _random_obstacles(None, obstacles_num, self.room_size, self.dynamics.arm)
        else:
            self.obstacles = None

        ################################################################################
        ## CONTROL
        if self.raw_control:
            if self.dim_mode == '1D':  # Z axis only
                self.controller = VerticalControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '2D':  # X and Z axes only
                self.controller = VertPlaneControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            elif self.dim_mode == '3D':
                self.controller = RawControl(self.dynamics, zero_action_middle=self.raw_control_zero_middle)
            else:
                raise ValueError('QuadEnv: Unknown dimensionality mode %s' % self.dim_mode)
        else:
            self.controller = NonlinearPositionController(self.dynamics, tf_control=self.tf_control)

        ################################################################################
        ## ACTIONS
        self.action_space = self.controller.action_space(self.dynamics)

        ################################################################################
        ## STATE VECTOR FUNCTION
        self.state_vector = getattr(get_state, "state_" + self.obs_repr)

    def make_observation_space(self):
        self.wall_offset = 0.3
        room_range = self.room_box[1] - self.room_box[0]
        room_max_dist = np.linalg.norm(self.room_box[1] - self.room_box[0]) * np.ones(1)
        self.obs_space_low_high = {
            "xyz": [-room_range, room_range],
            "xyzr": [-room_range, room_range],
            "vxyz": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "vxyzr": [-self.dynamics.vxyz_max * np.ones(3), self.dynamics.vxyz_max * np.ones(3)],
            "acc": [-self.dynamics.acc_max * np.ones(3), self.dynamics.acc_max * np.ones(3)],
            "R": [-np.ones(9), np.ones(9)],
            "omega": [-self.dynamics.omega_max * np.ones(3), self.dynamics.omega_max * np.ones(3)],
            "t2w": [0. * np.ones(1), 5. * np.ones(1)],
            "t2t": [0. * np.ones(1), 1. * np.ones(1)],
            "h": [0. * np.ones(1), self.room_box[1][2] * np.ones(1)],
            "act": [np.zeros(4), np.ones(4)],
            "quat": [-np.ones(4), np.ones(4)],
            "euler": [-np.pi * np.ones(3), np.pi * np.ones(3)],
            "rxyz": [-room_range, room_range], # rxyz stands for relative pos between quadrotors
            "rvxyz": [-2.0 * self.dynamics.vxyz_max * np.ones(3), 2.0 * self.dynamics.vxyz_max * np.ones(3)], # rvxyz stands for relative velocity between quadrotors
            "roxyz": [-room_range, room_range], # roxyz stands for relative pos between quadrotor and obstacle
            "rovxyz": [-20.0 * np.ones(3), 20.0 * np.ones(3)], # rovxyz stands for relative velocity between quadrotor and obstacle
            "osize": [np.zeros(3), 20.0 * np.ones(3)],  # obstacle size, [[0., 0., 0.], [20., 20., 20.]]
            "otype": [np.zeros(1), 20.0 * np.ones(1)],  # obstacle type, [[0.], [20.]], which means we can support 21 types of obstacles
            "goal": [-room_range, room_range],
            "nbr_dist": [np.zeros(1), room_max_dist],
            "nbr_goal_dist": [np.zeros(1), room_max_dist],
            "wall": [np.zeros(6), 5.0 * np.ones(6)],
        }
        self.obs_comp_names = list(self.obs_space_low_high.keys())
        self.obs_comp_sizes = [self.obs_space_low_high[name][1].size for name in self.obs_comp_names]

        obs_comps = self.obs_repr.split("_")
        if self.swarm_obs == 'pos_vel' and self.num_agents > 1:
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz']) * self.num_use_neighbor_obs
        elif self.swarm_obs == 'pos_vel_goals' and self.num_agents > 1:
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz'] + ['goal']) * self.num_use_neighbor_obs
        elif self.swarm_obs == 'pos_vel_goals_ndist_gdist' and self.num_agents > 1:
            obs_comps = obs_comps + (['rxyz'] + ['rvxyz'] + ['goal'] + ['nbr_dist'] + ['nbr_goal_dist']) * self.num_use_neighbor_obs
        if self.obstacle_mode != 'no_obstacles' and self.obstacle_num > 0:
            obs_comps = obs_comps + (['roxyz'] + ['rovxyz'] + ['osize'] + ['otype']) * self.obstacle_num

        print("Observation components:", obs_comps)
        obs_low, obs_high = [], []
        for comp in obs_comps:
            obs_low.append(self.obs_space_low_high[comp][0])
            obs_high.append(self.obs_space_low_high[comp][1])
        obs_low = np.concatenate(obs_low)
        obs_high = np.concatenate(obs_high)

        self.obs_comp_sizes_dict, self.obs_space_comp_indx, self.obs_comp_end = {}, {}, []
        end_indx = 0
        for obs_i, obs_name in enumerate(self.obs_comp_names):
            end_indx += self.obs_comp_sizes[obs_i]
            self.obs_comp_sizes_dict[obs_name] = self.obs_comp_sizes[obs_i]
            self.obs_space_comp_indx[obs_name] = obs_i
            self.obs_comp_end.append(end_indx)

        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)
        return self.observation_space

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        self.actions[1] = copy.deepcopy(self.actions[0])
        self.actions[0] = copy.deepcopy(action)
        # print('actions_norm: ', np.linalg.norm(self.actions[0]-self.actions[1]))

        # if not self.crashed:
        # print('goal: ', self.goal, 'goal_type: ', type(self.goal))
        self.controller.step_func(dynamics=self.dynamics,
                                  action=action,
                                  goal=self.goal,
                                  dt=self.dt,
                                  # observation=np.expand_dims(self.state_vector(self), axis=0))
                                  observation=None)  # assuming we aren't using observations in step function
        # self.oracle.step(self.dynamics, self.goal, self.dt)
        # self.scene.update_state(self.dynamics, self.goal)

        if self.obstacles is not None:
            self.crashed = self.obstacles.detect_collision(self.dynamics)
        else:
            self.crashed = self.dynamics.pos[2] <= self.dynamics.arm
        self.crashed = self.crashed or not np.array_equal(self.dynamics.pos,
                                                          np.clip(self.dynamics.pos,
                                                                  a_min=self.room_box[0],
                                                                  a_max=self.room_box[1]))

        self.time_remain = self.ep_len - self.tick
        reward, rew_info = compute_reward_weighted(self.dynamics, self.goal, action, self.dt, self.crashed,
                                                   self.time_remain,
                                                   rew_coeff=self.rew_coeff, action_prev=self.actions[1], quads_settle=self.quads_settle,
                                                   quads_settle_range_meters=self.quads_settle_range_meters,
                                                   quads_vel_reward_out_range=self.quads_vel_reward_out_range
        )
        self.tick += 1
        done = self.tick > self.ep_len  # or self.crashed
        sv = self.state_vector(self)

        self.traj_count += int(done)

        ## TODO: OPTIMIZATION: sv_comp should be a dictionary formed when state() function is called
        sv_comp = np.split(sv, self.obs_comp_end[:-1], axis=0)
        obs_comp = {
            "xyz": [self.dynamics.pos],
            "vxyz": [self.dynamics.vel],
            "acc": [self.dynamics.accelerometer],
            "omega": [self.dynamics.omega],
            "omega_dot": [self.dynamics.omega_dot],  # roll angular acceleration
            "R": [self.dynamics.rot.flatten()],
            "act": [action],
            "act_clipped": [np.clip(self.controller.action, a_min=0., a_max=1.)],
            "act_filtered": [self.dynamics.thrust_cmds_damp],
            "act_torque": [self.dynamics.prop_ccw * self.dynamics.thrust_cmds_damp],
            "torque": [self.dynamics.torque],
            "goal": [self.goal]
        }

        dyn_params = {
            "mass": [self.dynamics.mass],
            "motor_linearity": [self.dynamics.motor_linearity],
            "motor_time_up": [self.dynamics.motor_damp_time_up],
            "motor_time_down": [self.dynamics.motor_damp_time_down],
            "motor_assymetry": [self.dynamics.motor_assymetry],
            "motor_pos": [self.dynamics.prop_pos.flatten()],
            "motor_ccw": [self.dynamics.prop_ccw],
            "t2w": [self.dynamics.thrust_to_weight],
            "t2t": [self.dynamics.torque_to_thrust],
            "t2i": [self.dynamics.torque_to_inertia],
            "inertia": [self.dynamics.inertia],
            "thrust_max": [np.mean(self.dynamics.thrust_max)],
            "torque_max": [np.mean(self.dynamics.torque_max)],
            "arm": [self.dynamics.arm],
            "grav": [GRAV],
            "dt": [self.dt * self.sim_steps],
        }

        # print(sv, obs_comp, dyn_params, self.obs_comp_sizes)      
        return sv, reward, done, {'rewards': rew_info, "obs_comp": obs_comp, "dyn_params": dyn_params}

    def resample_dynamics(self):
        """
        Allows manual dynamics resampling when needed.
        WARNING: 
            - Randomization dyring an episode is not supported
            - MUST call reset() after this function
        """
        ## Getting base parameters (could also be random parameters)
        self.dynamics_params = self.dyn_base_sampler.sample()

        ## Now, updating if we are providing modifications
        if self.dynamics_change is not None:
            dict_update_existing(self.dynamics_params, self.dynamics_change)

        ## Applying sampler 1
        if self.dyn_sampler_1 is not None:
            self.dynamics_params = self.dyn_sampler_1.sample(self.dynamics_params)

        ## Applying sampler 2
        if self.dyn_sampler_2 is not None:
            self.dynamics_params = self.dyn_sampler_2.sample(self.dynamics_params)

        ## Checking that quad params make sense
        quad_rand.check_quad_param_limits(self.dynamics_params)

        ## Updating params
        self.update_dynamics(dynamics_params=self.dynamics_params)


    def _reset(self):
        ## I have to update state vector 
        ##############################################################
        ## DYNAMICS RANDOMIZATION AND UPDATE       
        if self.dynamics_randomize_every is not None and \
                (self.traj_count + 1) % (self.dynamics_randomize_every) == 0:
            self.resample_dynamics()

        ## CURRICULUM (NOT REALLY NEEDED ANYMORE)
        # from 0.5 to 10 after 100k episodes (a form of curriculum)
        if self.box < 10:
            self.box = self.box * self.box_scale
        x, y, z = self.np_random.uniform(-self.box, self.box, size=(3,)) + self.goal

        if self.dim_mode == '1D':
            x, y = self.goal[0], self.goal[1]
        elif self.dim_mode == '2D':
            y = self.goal[1]
        # Since being near the groud means crash we have to start above
        if z < 0.25: z = 0.25
        pos = npa(x, y, z)

        ##############################################################
        ## INIT STATE
        ## Initializing rotation and velocities
        if self.init_random_state:
            if self.dim_mode == '1D':
                omega, rotation = npa(0, 0, 0), np.eye(3)
                vel = np.array([0., 0., self.max_init_vel * np.random.rand()])
            elif self.dim_mode == '2D':
                omega = npa(0, self.max_init_omega * np.random.rand(), 0)
                vel = self.max_init_vel * np.random.rand(3)
                vel[1] = 0.
                theta = np.pi * np.random.rand()
                c, s = np.cos(theta), np.sin(theta)
                rotation = np.array(((c, 0., -s), (0., 1., 0.), (s, 0., c)))
            else:
                # It already sets the state internally
                _, vel, rotation, omega = self.dynamics.random_state(
                    box=(self.room_length, self.room_width, self.room_height), vel_max=self.max_init_vel, omega_max=self.max_init_omega
                )
        else:
            ## INIT HORIZONTALLY WITH 0 VEL and OMEGA
            vel, omega = npa(0, 0, 0), npa(0, 0, 0)

            if self.dim_mode == '1D' or self.dim_mode == '2D':
                rotation = np.eye(3)
            else:
                # make sure we're sort of pointing towards goal (for mellinger controller)
                rotation = randyaw()
                while np.dot(rotation[:, 0], to_xyhat(-pos)) < 0.5:
                    rotation = randyaw()

        # Setting the generated state
        # print("QuadEnv: init: pos/vel/rot/omega:", pos, vel, rotation, omega)
        self.init_state = [pos, vel, rotation, omega]
        self.dynamics.set_state(pos, vel, rotation, omega)
        self.dynamics.reset()

        # Reseting some internal state (counters, etc)
        self.crashed = False
        self.tick = 0
        self.actions = [np.zeros([4, ]), np.zeros([4, ])]

        state = self.state_vector(self)
        return state

    def reset(self):
        return self._reset()

    def render(self, mode='human', **kwargs):
        """This class is only meant to be used as a component of QuadMultiEnv."""
        raise NotImplementedError()

    def step(self, action):
        return self._step(action)


class DummyPolicy(object):
    def __init__(self, dt=0.01, switch_time=2.5):
        self.action = np.zeros([4, ])
        self.dt = 0.

    def step(self, x):
        return self.action

    def reset(self):
        pass


class UpDownPolicy(object):
    def __init__(self, dt=0.01, switch_time=2.5):
        self.t = 0
        self.dt = dt
        self.switch_time = switch_time
        self.action_up = np.ones([4, ])
        self.action_up[:2] = 0.
        self.action_down = np.zeros([4, ])
        self.action_down[:2] = 1.

    def step(self, x):
        self.t += self.dt
        if self.t < self.switch_time:
            return self.action_up
        else:
            return self.action_down

    def reset(self):
        self.t = 0.


def test_rollout(quad, dyn_randomize_every=None, dyn_randomization_ratio=None,
                 render=True, traj_num=10, plot_step=None, plot_dyn_change=True, plot_thrusts=False,
                 sense_noise=None, policy_type="mellinger", init_random_state=False, obs_repr="xyz_vxyz_rot_omega",
                 csv_filename=None):
    import tqdm
    #############################
    # Init plottting
    if plot_step is not None:
        fig = plt.figure(1)
        # ax = plt.subplot(111)
        plt.show(block=False)

    # render = True
    # plot_step = 50
    time_limit = 25
    render_each = 2
    rollouts_num = traj_num
    plot_obs = False

    if policy_type == "mellinger":
        raw_control = False
        raw_control_zero_middle = True
        policy = DummyPolicy()  # since internal Mellinger takes care of the policy
    elif policy_type == "updown":
        raw_control = True
        raw_control_zero_middle = False
        policy = UpDownPolicy()

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = {
            "class": "RelativeSampler",
            "noise_ratio": dyn_randomization_ratio,
            "sampler": "normal"
        }

    env = QuadrotorSingle(dynamics_params=quad, raw_control=raw_control,
                          raw_control_zero_middle=raw_control_zero_middle,
                          dynamics_randomize_every=dyn_randomize_every, dyn_sampler_1=sampler_1,
                          sense_noise=sense_noise, init_random_state=init_random_state, obs_repr=obs_repr)

    policy.dt = 1. / env.control_freq

    env.max_episode_steps = time_limit
    print('Reseting env ...')
    print("Obs repr: ", env.obs_repr)
    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high, "size:",
              env.observation_space.high.size)
        print('Action space:', env.action_space.low, env.action_space.high, "size:", env.observation_space.high.size)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high,
              "size:", env.observation_space[0].spaces[0].high.size)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high, "size:",
              env.action_space[0].spaces[0].high.size)
    # input('Press any key to continue ...')

    ## Collected statistics for dynamics
    dyn_param_names = [
        "mass",
        "inertia",
        "thrust_to_weight",
        "torque_to_thrust",
        "thrust_noise_ratio",
        "vel_damp",
        "damp_omega_quadratic",
        "torque_to_inertia"
    ]

    dyn_param_stats = [[] for i in dyn_param_names]

    action = np.array([0.0, 0.5, 0.0, 0.5])
    rollouts_id = 0

    start_time = time.time()
    # while rollouts_id < rollouts_num:
    for rollouts_id in tqdm.tqdm(range(rollouts_num)):
        rollouts_id += 1
        s = env.reset()
        policy.reset()
        ## Diagnostics
        observations = []
        velocities = []
        actions = []
        thrusts = []
        csv_data = []

        ## Collecting dynamics params
        if plot_dyn_change:
            for par_i, par in enumerate(dyn_param_names):
                dyn_param_stats[par_i].append(np.array(getattr(env.dynamics, par)).flatten())
                # print(par, dyn_param_stats[par_i][-1])

        t = 0
        while True:
            if render and (t % render_each == 0): env.render()
            action = policy.step(s)
            s, r, done, info = env.step(action)

            actions.append(action)
            thrusts.append(env.dynamics.thrust_cmds_damp)
            observations.append(s)
            # print('Step: ', t, ' Obs:', s)
            quat = R2quat(rot=s[6:15])
            csv_data.append(np.concatenate([np.array([1.0 / env.control_freq * t]), s[0:3], quat]))

            if plot_step is not None and t % plot_step == 0:
                plt.clf()

                if plot_obs:
                    observations_arr = np.array(observations)
                    # print('observations array shape', observations_arr.shape)
                    dimenstions = observations_arr.shape[1]
                    for dim in range(dimenstions):
                        plt.plot(observations_arr[:, dim])
                    plt.legend([str(x) for x in range(observations_arr.shape[1])])

                plt.pause(0.05)  # have to pause otherwise does not draw
                plt.draw()

            if done: break
            t += 1

        if plot_thrusts:
            plt.figure(3, figsize=(10, 10))
            ep_time = np.linspace(0, policy.dt * len(actions), len(actions))
            actions = np.array(actions)
            thrusts = np.array(thrusts)
            for i in range(4):
                plt.plot(ep_time, actions[:, i], label="Thrust desired %d" % i)
                plt.plot(ep_time, thrusts[:, i], label="Thrust produced %d" % i)
            plt.legend()
            plt.show(block=False)
            input("Press Enter to continue...")

        if csv_filename is not None:
            import csv
            with open(csv_filename, mode="w") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for row in csv_data:
                    csv_writer.writerow([i for i in row])

    if plot_dyn_change:
        dyn_par_normvar = []
        dyn_par_means = []
        dyn_par_var = []
        plt.figure(2, figsize=(10, 10))
        for par_i, par in enumerate(dyn_param_stats):
            plt.subplot(3, 3, par_i + 1)
            par = np.array(par)

            ## Compute stats
            # print(dyn_param_names[par_i], par)
            dyn_par_means.append(np.mean(par, axis=0))
            dyn_par_var.append(np.std(par, axis=0))
            dyn_par_normvar.append(dyn_par_var[-1] / dyn_par_means[-1])

            if par.shape[1] > 1:
                for vi in range(par.shape[1]):
                    plt.plot(par[:, vi])
            else:
                plt.plot(par)
            # plt.title(dyn_param_names[par_i] + "\n Normvar: %s" % str(dyn_par_normvar[-1]))
            plt.title(dyn_param_names[par_i])
            print(dyn_param_names[par_i], "NormVar: ", dyn_par_normvar[-1])

    print("##############################################################")
    print("Total time: ", time.time() - start_time)

    # print('Rollouts are done!')
    # plt.pause(2.0)
    # plt.waitforbuttonpress()
    if plot_step is not None or plot_dyn_change:
        plt.show(block=False)
        input("Press Enter to continue...")


def benchmark(quad, dyn_randomize_every=None, dyn_randomization_ratio=None,
              render=True, traj_num=10, plot_step=None, plot_dyn_change=True, plot_thrusts=False,
              sense_noise=None, policy_type="mellinger", init_random_state=False, obs_repr="xyz_vxyz_rot_omega",
              csv_filename=None):
    import tqdm
    rollouts_num = traj_num

    if policy_type == "mellinger":
        raw_control = False
        raw_control_zero_middle = True
        policy = DummyPolicy()  # since internal Mellinger takes care of the policy
    elif policy_type == "updown":
        raw_control = True
        raw_control_zero_middle = False
        policy = UpDownPolicy()

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = {
            "class": "RelativeSampler",
            "noise_ratio": dyn_randomization_ratio,
            "sampler": "normal"
        }

    env = QuadrotorSingle(dynamics_params=quad, raw_control=raw_control,
                          raw_control_zero_middle=raw_control_zero_middle,
                          dynamics_randomize_every=dyn_randomize_every, dyn_sampler_1=sampler_1,
                          sense_noise=sense_noise, init_random_state=init_random_state, obs_repr=obs_repr)

    policy.dt = 1. / env.control_freq
    render = False
    render_each = 2

    print('Reseting env ...')
    print("Obs repr: ", env.obs_repr)
    try:
        print('Observation space:', env.observation_space.low, env.observation_space.high, "size:",
              env.observation_space.high.size)
        print('Action space:', env.action_space.low, env.action_space.high, "size:", env.observation_space.high.size)
    except:
        print('Observation space:', env.observation_space.spaces[0].low, env.observation_space[0].spaces[0].high,
              "size:", env.observation_space[0].spaces[0].high.size)
        print('Action space:', env.action_space[0].spaces[0].low, env.action_space[0].spaces[0].high, "size:",
              env.action_space[0].spaces[0].high.size)

    ## Collected statistics for dynamics
    dyn_param_names = [
        "mass",
        "inertia",
        "thrust_to_weight",
        "torque_to_thrust",
        "thrust_noise_ratio",
        "vel_damp",
        "damp_omega_quadratic",
        "torque_to_inertia"
    ]

    dyn_param_stats = [[] for i in dyn_param_names]

    start_time = time.time()
    for rollouts_id in tqdm.tqdm(range(rollouts_num)):
        s = env.reset()
        policy.reset()

        t = 0
        while True:
            if render and (t % render_each == 0): env.render()
            action = policy.step(s)
            s, r, done, info = env.step(action)
            if done: break

    print("##############################################################")
    print("Total time: ", time.time() - start_time)
    input("Press Enter to continue...")


def parse_quad_args(argv):
    # parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-m', "--mode",
        default="mellinger",
        help="Test mode: "
             "mellinger - rollout with mellinger controller"
             "updown - rollout with UpDown controller (to test step responses)"
    )
    parser.add_argument(
        '-q', "--quad",
        default="DefaultQuad",
        help="Quadrotor model to use: \n" +
             "- DefaultQuad \n" +
             "- Crazyflie \n" +
             "- MediumQuad \n" +
             "- RandomQuad"
    )
    parser.add_argument(
        '-dre', "--dyn_randomize_every",
        type=int,
        help="How often (in terms of trajectories) to perform randomization"
    )
    parser.add_argument(
        '-drr', "--dyn_randomization_ratio",
        type=float,
        default=None,
        help="Randomization ratio for random sampling of dynamics parameters"
    )
    parser.add_argument(
        '-r', "--render",
        action="store_false",
        help="Use this flag to turn off rendering"
    )
    parser.add_argument(
        '-trj', "--traj_num",
        type=int,
        default=10,
        help="Number of trajectories to run"
    )
    parser.add_argument(
        '-plt', "--plot_step",
        type=int,
        help="Plot step"
    )
    parser.add_argument(
        '-pltdyn', "--plot_dyn_change",
        action="store_true",
        help="Plot the dynamics change from trajectory to trajectory?"
    )
    parser.add_argument(
        '-pltact', "--plot_actions",
        action="store_true",
        help="Plot actions commanded and thrusts produced after damping"
    )
    parser.add_argument(
        '-sn', "--sense_noise",
        action="store_false",
        help="Add sensor noise? Use this flag to turn the noise off"
    )
    parser.add_argument(
        '-irs', "--init_random_state",
        action="store_true",
        help="Add sensor noise?"
    )
    parser.add_argument(
        '-csv', "--csv_filename",
        help="Filename for qudrotor data"
    )
    parser.add_argument(
        '-o', "--obs_repr",
        default="xyz_vxyz_R_omega_act",
        help="State components. Options:\n" +
             "xyz_vxyz_R_omega" +
             "xyz_vxyz_R_omega_act" +
             "xyz_vxyz_R_omega_acc_act"
    )
    parser.add_argument(
        '-b', "--benchmark",
        action="store_true",
        help="Simple benchmark, i.e. running time"
    )

    args = parser.parse_args(args=argv)
    return args


def main(argv):
    args = parse_quad_args(argv)

    if args.sense_noise:
        sense_noise = "default"
    else:
        sense_noise = None

    if args.benchmark:
        print('Running benchmark ...')
        benchmark(
            quad=args.quad,
            dyn_randomize_every=args.dyn_randomize_every,
            dyn_randomization_ratio=args.dyn_randomization_ratio,
            render=args.render,
            traj_num=args.traj_num,
            plot_step=args.plot_step,
            plot_dyn_change=args.plot_dyn_change,
            plot_thrusts=args.plot_actions,
            sense_noise=sense_noise,
            policy_type=args.mode,
            init_random_state=args.init_random_state,
            obs_repr=args.obs_repr,
            csv_filename=args.csv_filename,
        )
    else:
        print('Running test rollout ...')
        test_rollout(
            quad=args.quad,
            dyn_randomize_every=args.dyn_randomize_every,
            dyn_randomization_ratio=args.dyn_randomization_ratio,
            render=args.render,
            traj_num=args.traj_num,
            plot_step=args.plot_step,
            plot_dyn_change=args.plot_dyn_change,
            plot_thrusts=args.plot_actions,
            sense_noise=sense_noise,
            policy_type=args.mode,
            init_random_state=args.init_random_state,
            obs_repr=args.obs_repr,
            csv_filename=args.csv_filename,
        )


@njit
def calculate_torque_integrate_rotations_and_update_omega(thrust_cmds, dt, eps, motor_damp_time_up,
                                                          motor_damp_time_down, thrust_cmds_damp,
                                                          thrust_rot_damp, thr_noise, thrust_max, motor_linearity,
                                                          prop_crossproducts, prop_ccw, torque_max, rot, omega,
                                                          eye, since_last_svd, since_last_svd_limit, inertia,
                                                          damp_omega_quadratic, omega_max, pos, vel):
    # Filtering the thruster and adding noise
    thrust_cmds = np.clip(thrust_cmds, 0., 1.)
    motor_tau_up = 4 * dt / (motor_damp_time_up + eps)
    motor_tau_down = 4 * dt / (motor_damp_time_down + eps)
    motor_tau = motor_tau_up * np.ones(4)
    motor_tau[thrust_cmds < thrust_cmds_damp] = motor_tau_down
    motor_tau[motor_tau > 1.] = 1.

    # Since NN commands thrusts we need to convert to rot vel and back
    thrust_rot = thrust_cmds ** 0.5
    thrust_rot_damp = motor_tau * (thrust_rot - thrust_rot_damp) + thrust_rot_damp
    thrust_cmds_damp = thrust_rot_damp ** 2

    # Adding noise
    thrust_noise = thrust_cmds * thr_noise
    thrust_cmds_damp = np.clip(thrust_cmds_damp + thrust_noise, 0.0, 1.0)
    thrusts = thrust_max * angvel2thrust_numba(thrust_cmds_damp, motor_linearity)

    # Prop cross-product gives torque directions
    torques = prop_crossproducts * np.reshape(thrusts, (-1, 1))

    # Additional torques along z-axis caused by propeller rotations
    torques[:, 2] += torque_max * prop_ccw * thrust_cmds_damp

    # Net torque: sum over propellers
    thrust_torque = np.sum(torques, 0)

    # Rotor drag and Rolling forces and moments
    rotor_visc_torque = rotor_drag_force = np.zeros(3)

    # (Square) Damping using torques (in case we would like to add damping using torques)
    torque = thrust_torque + rotor_visc_torque
    thrust = np.array([0, 0, np.sum(thrusts)])

    # ROTATIONAL DYNAMICS
    # Integrating rotations (based on current values)
    omega_vec = rot @ omega
    wx, wy, wz = omega_vec
    omega_norm = np.linalg.norm(omega_vec)
    if omega_norm != 0:
        K = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]]) / omega_norm
        rot_angle = omega_norm * dt
        dRdt = eye + np.sin(rot_angle) * K + (1. - np.cos(rot_angle)) * (K @ K)
        rot = dRdt @ rot

    # SVD is not strictly required anymore. Performing it rarely, just in case
    since_last_svd += dt
    if since_last_svd > since_last_svd_limit:
        u, s, v = np.linalg.svd(rot)
        rot = u @ v
        since_last_svd = 0

    # COMPUTING OMEGA UPDATE
    # Linear damping
    omega_dot = ((1.0 / inertia) * (numba_cross(-omega, inertia * omega) + torque))

    # Quadratic damping
    omega_damp_quadratic = np.clip(damp_omega_quadratic * omega ** 2, 0.0, 1.0)
    omega = omega + (1.0 - omega_damp_quadratic) * dt * omega_dot
    omega = np.clip(omega, -omega_max, omega_max)

    # Computing position
    pos = pos + dt * vel

    return motor_tau_up, motor_tau_down, thrust_rot_damp, thrust_cmds_damp, torques, \
           torque, rot, since_last_svd, omega_dot, omega, pos, thrust, rotor_drag_force


@njit
def compute_velocity_and_acceleration(vel, grav_cnst_arr, mass, rot, sum_thr_drag, vel_damp, dt, rot_tpose,
                                      grav_arr):
    # Computing accelerations
    acc = grav_cnst_arr + ((1.0 / mass) * (rot @ sum_thr_drag))

    # Computing velocities
    vel = (1.0 - vel_damp) * vel + dt * acc

    # Accelerometer measures so called "proper acceleration" that includes gravity with the opposite sign
    accm = rot_tpose @ (acc + grav_arr)
    return vel, acc, accm


if __name__ == '__main__':
    main(sys.argv)
