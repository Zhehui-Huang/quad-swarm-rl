import numpy as np


# NOTE: the state_* methods are static because otherwise getattr memorizes self

def _policy_frame_position_velocity(self, pos, vel, rot=None):
    relative_pos = np.array(pos - self.goal[:3], copy=True)
    policy_vel = np.array(vel, copy=True)

    if self.control_type == "velocity_yaw_body_avoid":
        observed_rot = self.dynamics.rot if rot is None else rot
        body_x = observed_rot[:2, 0]
        norm = np.linalg.norm(body_x)
        if norm < 1e-6:
            world_to_body = np.eye(2)
        else:
            c, s = body_x[0] / norm, body_x[1] / norm
            world_to_body = np.array([[c, s], [-s, c]])

        relative_pos[:2] = world_to_body @ relative_pos[:2]
        policy_vel[:2] = world_to_body @ policy_vel[:2]

    return relative_pos, policy_vel


def state_xyz_vxyz_R_omega(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    relative_pos, policy_vel = _policy_frame_position_velocity(self, pos, vel, rot)
    return np.concatenate([relative_pos, policy_vel, rot.flatten(), omega])


def state_xyz_vxyz_R_omega_floor(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    relative_pos, policy_vel = _policy_frame_position_velocity(self, pos, vel, rot)
    return np.concatenate([relative_pos, policy_vel, rot.flatten(), omega, (pos[2],)])


def state_xyz_vxyz_R_omega_wall(self):
    if self.use_numba:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise_numba(
            self.dynamics.pos,
            self.dynamics.vel,
            self.dynamics.rot,
            self.dynamics.omega,
            self.dynamics.accelerometer,
            self.dt
        )
    else:
        pos, vel, rot, omega, acc = self.sense_noise.add_noise(
            pos=self.dynamics.pos,
            vel=self.dynamics.vel,
            rot=self.dynamics.rot,
            omega=self.dynamics.omega,
            acc=self.dynamics.accelerometer,
            dt=self.dt
        )
    relative_pos, policy_vel = _policy_frame_position_velocity(self, pos, vel, rot)
    wall_box_0 = np.clip(pos - self.room_box[0], a_min=0.0, a_max=5.0)
    wall_box_1 = np.clip(self.room_box[1] - pos, a_min=0.0, a_max=5.0)
    return np.concatenate([relative_pos, policy_vel, rot.flatten(), omega, wall_box_0, wall_box_1])
