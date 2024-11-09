import numpy as np
from scipy.spatial.transform import Rotation as scipy_rotation

# NOTE: the state_* methods are static because otherwise getattr memorizes self

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

    if self.dynamic_goal:
        goal_rot = scipy_rotation.from_euler('z', [self.goal[12]])
        goal_rot = goal_rot.as_matrix()
        return np.concatenate([pos - self.goal[:3], vel - self.goal[3:6], rot.flatten() - goal_rot.flatten(), omega - self.goal[9:12]])
    else:
        if self.obs_rel_rot:
            return np.concatenate([pos - self.goal[:3], vel, rot.flatten() - self.base_rot.flatten(), omega])
        else:
            return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega])


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
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])


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
    # return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, (pos[2],)])
    wall_box_0 = np.clip(pos - self.room_box[0], a_min=0.0, a_max=5.0)
    wall_box_1 = np.clip(self.room_box[1] - pos, a_min=0.0, a_max=5.0)
    return np.concatenate([pos - self.goal[:3], vel, rot.flatten(), omega, wall_box_0, wall_box_1])
