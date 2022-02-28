import copy

import numpy as np

from gym_art.quadrotor_multi.quad_obstacle_utils import OBSTACLES_SHAPE_LIST, STATIC_OBSTACLE_DOOR
from gym_art.quadrotor_multi.params import quad_arm_size


EPS = 1e-6
GRAV = 9.81  # default gravitational constant
TRAJ_LIST = ['gravity', 'electron']


class SingleObstacle:
    def __init__(self, max_init_vel=1., init_box=2.0, mode='no_obstacles', shape='sphere', size=0.0, quad_size=0.04,
                 dt=0.05, traj='gravity', obs_mode='relative', index=0, obs_type='pos_size', all_pos_arr=None,
                 inf_height=False, room_dims=(10.0, 10.0, 10.0), rel_pos_mode=0, rel_pos_clip_value=2.0):
        if all_pos_arr is None:
            all_pos_arr = []
        self.max_init_vel = max_init_vel
        self.init_box = init_box  # means the size of initial space that the obstacles spawn at
        self.mode = mode
        self.shape = shape
        self.origin_size = size
        self.size = size  # sphere: diameter, cube: edge length; cylinder: diameter
        self.quad_size = quad_size
        self.dt = dt
        self.traj = traj
        self.tmp_traj = traj
        self.pos = np.array([5., 5., -5.])
        self.vel = np.array([0., 0., 0.])
        self.formation_size = 0.0
        self.goal_central = np.array([0., 0., 2.])
        self.shape_list = OBSTACLES_SHAPE_LIST
        self.obs_mode = obs_mode
        self.index = index
        self.obs_type = obs_type
        self.all_pos_arr = all_pos_arr
        self.inf_height = inf_height
        self.room_dims = room_dims
        self.rel_pos_mode = rel_pos_mode
        self.rel_pos_clip_value = rel_pos_clip_value
        self.inside_room_box_flag = True

    def reset(self, set_obstacle=None, formation_size=0.0, goal_central=np.array([0., 0., 2.]), shape='sphere',
              quads_pos=None, quads_vel=None, new_pos=None, inside_room_box_flag=True):
        self.inside_room_box_flag = inside_room_box_flag
        if self.inside_room_box_flag:
            self.size = self.origin_size
        else:
            self.size = 1e-6


        # Initial position and velocity
        if set_obstacle is None:
            raise ValueError('set_obstacle is None')

        self.formation_size = formation_size
        self.goal_central = goal_central

        # Reset shape and size
        self.shape = shape
        if 'fixsize' not in self.mode:
            self.size = np.random.uniform(low=0.15, high=0.5)

        if set_obstacle:
            if self.mode.startswith('static'):
                self.static_obstacle(new_pos=new_pos)
            elif self.mode == 'dynamic':
                if self.traj == "mix":
                    traj_id = np.random.randint(low=0, high=len(TRAJ_LIST))
                    self.tmp_traj = TRAJ_LIST[traj_id]
                else:
                    self.tmp_traj = self.traj

                if self.tmp_traj == "electron":
                    self.dynamic_obstacle_electron()
                elif self.tmp_traj == "gravity":
                    # Try 1 + 100 times, make sure initial vel, both vx and vy < 3.0
                    self.dynamic_obstacle_grav()
                    for _ in range(100):
                        if abs(self.vel[0]) > 3.0 or abs(self.vel[1]) > 3.0:
                            self.dynamic_obstacle_grav()
                        else:
                            break
                else:
                    raise NotImplementedError(f'{self.traj} not supported!')
            else:
                raise NotImplementedError(f'{self.mode} not supported!')
        else:
            self.pos = np.array([5., 5., -5.])
            self.vel = np.array([0., 0., 0.])

        obs = self.update_obs(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
        return obs

    def static_obstacle(self, new_pos=None):
        # Init position for an obstacle
        self.vel = np.array([0., 0., 0.])
        if 'static_door' in self.mode:
            self.pos = STATIC_OBSTACLE_DOOR[self.index]
        elif 'static_random_place' in self.mode:
            self.pos = self.all_pos_arr[self.index]
        elif 'static_pillar' in self.mode:
            if new_pos is None:
                self.pos = self.all_pos_arr[self.index]
            else:
                self.pos = new_pos
        else:
            raise NotImplementedError(f'{self.mode} is not supported!')

    def dynamic_obstacle_grav(self):
        # Init position for an obstacle
        x = np.random.uniform(low=-self.init_box, high=self.init_box)
        y = np.random.uniform(low=0.67 * x, high=1.5 * x)
        sign_y = np.random.uniform(low=0.0, high=1.0)
        if sign_y < 0.5:
            y = -y

        z = np.random.uniform(low=-0.5 * self.init_box, high=0.5 * self.init_box) + self.goal_central[2]
        z = max(self.size / 2 + 0.5, z)

        # Make the position of obstacles out of the space of goals
        formation_range = self.formation_size + self.size / 2
        formation_range = max(formation_range, 0.5)
        rel_x = abs(x) - formation_range
        rel_y = abs(y) - formation_range
        if rel_x <= 0:
            x += np.sign(x) * np.random.uniform(low=abs(rel_x) + 0.5,
                                                high=abs(rel_x) + 1.0)
        if rel_y <= 0:
            y += np.sign(y) * np.random.uniform(low=abs(rel_y) + 0.5,
                                                high=abs(rel_y) + 1.0)
        self.pos = np.array([x, y, z])

        # Init velocity for an obstacle
        # obstacle_vel = np.random.uniform(low=-self.max_init_vel, high=self.max_init_vel, size=(3,))
        self.vel = self.get_grav_init_vel()

    def dynamic_obstacle_electron(self):
        # Init position for an obstacle
        x, y = np.random.uniform(-self.init_box, self.init_box, size=(2,))
        z = np.random.uniform(low=-0.5 * self.init_box, high=0.5 * self.init_box) + self.goal_central[2]
        z = max(self.size / 2 + 0.5, z)

        # Make the position of obstacles out of the space of goals
        formation_range = self.formation_size + self.size / 2
        rel_x = abs(x) - formation_range
        rel_y = abs(y) - formation_range
        if rel_x <= 0:
            x += np.sign(x) * np.random.uniform(low=abs(rel_x) + 0.5,
                                                high=abs(rel_x) + 1.0)
        if rel_y <= 0:
            y += np.sign(y) * np.random.uniform(low=abs(rel_y) + 0.5,
                                                high=abs(rel_y) + 1.0)
        self.pos = np.array([x, y, z])

        # Init velocity for an obstacle
        self.vel = self.get_electron_init_vel()

    def get_grav_init_vel(self):
        # Calculate the initial position of the obstacle, which can make it finally fly through the center of the
        # goal formation.
        # There are three situations for the initial positions
        # 1. Below the center of goals (dz > 0). Then, there are two trajectories.
        # 2. Equal or above the center of goals (dz <= 0). Then, there is only one trajectory.
        # More details, look at: https://drive.google.com/file/d/1Vp0TaiQ_4vN9pH-Z3uGR54gNx6jh9thP/view
        target_noise = np.random.uniform(-0.2, 0.2, size=(3,))
        target_pos = self.goal_central + target_noise
        dx, dy, dz = target_pos - self.pos

        vz_noise = np.random.uniform(low=0.0, high=1.0)
        vz = np.sqrt(2 * GRAV * abs(dz)) + vz_noise
        delta = np.sqrt(vz * vz - 2 * GRAV * dz)
        if dz > 0:
            # t_list = [(vz + delta) / GRAV, (vz - delta) / GRAV]
            # t_index = round(np.random.uniform(low=0, high=1))
            # t = t_list[t_index]
            t = (vz + delta) / GRAV
        elif dz < 0:
            # vz_index = 0, vz < 0; vz_index = 1, vz > 0;
            # vz_index = round(np.random.uniform(low=0, high=1))
            # if vz_index == 0:  # vz < 0
            #     vz = - vz

            t = (vz + delta) / GRAV
        else:  # dz = 0, vz > 0
            vz = np.random.uniform(low=0.5 * self.max_init_vel, high=self.max_init_vel)
            t = 2 * vz / GRAV

        # Calculate vx
        vx = dx / t
        vy = dy / t
        vel = np.array([vx, vy, vz])
        return vel

    def get_electron_init_vel(self):
        vel_direct = self.goal_central - self.pos
        vel_direct_noise = np.random.uniform(low=-0.1, high=0.1, size=(3,))
        vel_direct += vel_direct_noise
        vel_magn = np.random.uniform(low=0., high=self.max_init_vel)
        vel = vel_magn * vel_direct / (np.linalg.norm(vel_direct) + EPS)
        return vel

    def update_obs(self, quads_pos, quads_vel, set_obstacle):
        obs = None
        if 'dynamic' in self.mode:
            # Add rel_pos, rel_vel, size, shape to obs, shape: num_agents * 10
            if (not set_obstacle) and self.obs_mode == 'absolute':
                rel_pos = self.pos - np.zeros((len(quads_pos), 3))
                rel_vel = self.vel - np.zeros((len(quads_pos), 3))
                obst_size = np.zeros((len(quads_pos), 1))
                obst_shape = np.zeros((len(quads_pos), 1))
            elif (not set_obstacle) and self.obs_mode == 'half_relative':
                rel_pos = self.pos - np.zeros((len(quads_pos), 3))
                rel_vel = self.vel - np.zeros((len(quads_pos), 3))
                obst_size = (self.size / 2) * np.ones((len(quads_pos), 1))
                obst_shape = self.shape_list.index(self.shape) * np.ones((len(quads_pos), 1))
            else:  # False, relative; True
                rel_pos = self.pos - quads_pos
                if self.rel_pos_mode == 1:
                    rel_dist = np.linalg.norm(rel_pos, axis=1)
                    rel_dist = np.maximum(rel_dist, 1e-6)
                    rel_pos_unit = rel_pos / rel_dist[:, None]
                    rel_pos -= rel_pos_unit * (0.5 * self.size + quad_arm_size)

                rel_vel = self.vel - quads_vel
                # obst_size: in xyz axis: radius for sphere, half edge length for cube
                obst_size = (self.size / 2) * np.ones((len(quads_pos), 1))
                obst_shape = self.shape_list.index(self.shape) * np.ones((len(quads_pos), 1))
            obs = np.concatenate((rel_pos, rel_vel, obst_size, obst_shape), axis=1)
        elif 'static' in self.mode:
            if self.inf_height or 'posxy' in self.obs_type:
                if self.obs_type == 'pos_vel_size':
                    rel_pos = self.pos - quads_pos
                    rel_pos[:, 2] = 0.0
                else:
                    rel_pos = self.pos[:2] - quads_pos[:, :2]
            else:
                rel_pos = self.pos - quads_pos

            if self.rel_pos_mode == 1:
                rel_dist = np.linalg.norm(rel_pos, axis=1)
                rel_dist = np.maximum(rel_dist, 1e-6)
                rel_pos_unit = rel_pos / rel_dist[:, None]
                rel_pos -= rel_pos_unit * (0.5 * self.size + quad_arm_size)
            elif self.rel_pos_mode == 2:
                rel_pos = np.clip(rel_pos, a_min=-1.0 * self.rel_pos_clip_value, a_max=self.rel_pos_clip_value)

            rel_vel = self.vel - quads_vel

            # obst_size: in xyz axis: radius for sphere, half edge length for cube
            obst_size = (self.size / 2) * np.ones((len(quads_pos), 1))
            obst_shape = self.shape_list.index(self.shape) * np.ones((len(quads_pos), 1))
            if self.obs_type == 'cpoint':
                closest_points = self.get_closest_points(quads_pos)
                if self.shape == 'cylinder' and self.inf_height:
                    obs = copy.deepcopy(closest_points)
                else:
                    obs = closest_points - quads_pos
            elif self.obs_type == 'pos_size' or self.obs_type == 'posxy_size':
                obs = np.concatenate((rel_pos, obst_size), axis=1)
            elif self.obs_type == 'pos_vel_size':
                rot_matrix = np.identity(3).reshape(-1)
                rot_matrix = np.tile(rot_matrix, rel_pos.shape[0]).reshape(rel_pos.shape[0], -1)
                omega = np.array([0., 0., 1.])
                omega = np.tile(omega, rel_pos.shape[0]).reshape(rel_pos.shape[0], -1)
                obs = np.concatenate((rel_pos, rel_vel, rot_matrix, omega, obst_size), axis=1)
            elif self.obs_type == 'pos_vel_size_shape':
                obs = np.concatenate((rel_pos, rel_vel, obst_size, obst_shape), axis=1)
            else:
                raise NotImplementedError(f'{self.obs_type} is not supported!')
        else:
            raise NotImplementedError(f'{self.obs_type} is not supported!')

        return obs

    def step(self, quads_pos=None, quads_vel=None, set_obstacle=None):
        if set_obstacle is None:
            raise ValueError('set_obstacle is None')

        if not set_obstacle or self.mode.startswith('static'):  # dynamic obstacles in the fly
            obs = self.update_obs(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
            return obs

        if self.tmp_traj == 'electron':
            obs = self.step_electron(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
            return obs
        elif self.tmp_traj == 'gravity':
            obs = self.step_gravity(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
            return obs
        else:
            raise NotImplementedError()

    def step_electron(self, quads_pos, quads_vel, set_obstacle):
        # Generate force, mimic force between electron, F = k*q1*q2 / r^2,
        # Here, F = r^2, k = 1, q1 = q2 = 1
        force_pos = 2 * self.goal_central - self.pos
        rel_force_goal = force_pos - self.goal_central
        force_noise = np.random.uniform(low=-0.5 * rel_force_goal, high=0.5 * rel_force_goal)
        force_pos = force_pos + force_noise
        rel_force_obstacle = force_pos - self.pos

        force = rel_force_obstacle
        # Calculate acceleration, F = ma, here, m = 1.0
        acc = force
        # Calculate position and velocity
        self.vel += self.dt * acc
        self.pos += self.dt * self.vel

        obs = self.update_obs(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
        return obs

    def step_gravity(self, quads_pos, quads_vel, set_obstacle):
        acc = np.array([0., 0., -GRAV])  # 9.81
        # Calculate velocity
        self.vel += self.dt * acc
        self.pos += self.dt * self.vel

        obs = self.update_obs(quads_pos=quads_pos, quads_vel=quads_vel, set_obstacle=set_obstacle)
        return obs

    def cube_detection(self, pos_quads=None):
        # https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
        # Sphere vs. AABB (Cuboid, not only cube)
        closest_poses = self.get_closest_points(pos_quads)
        # dist_arr means the distance between from drones to the closest point on the obstacle
        dist_arr = np.linalg.norm(pos_quads - closest_poses, axis=1)
        collision_arr = (dist_arr <= self.quad_size).astype(np.float32)

        return collision_arr, dist_arr

    def sphere_detection(self, pos_quads=None):
        # dist_arr means the distance between from drones to the closest point on the obstacle
        dist_arr = np.linalg.norm(pos_quads - self.pos, axis=1) - 0.5 * self.size
        collision_arr = (dist_arr <= self.quad_size).astype(np.float32)
        return collision_arr, dist_arr

    def cylinder_detection(self, pos_quads=None):
        # dist_arr means the distance between from drones to the closest point on the obstacle
        obst_radius = 0.5 * self.size
        delta_xy_dist = np.linalg.norm(pos_quads[:, :2] - self.pos[:2], axis=1) - obst_radius

        if self.inf_height:
            dist_arr = delta_xy_dist
        else:
            obst_half_height = 0.5 * self.size

            delta_xy_dist = np.maximum(delta_xy_dist, 0.0)
            # np.maximum(obst_min_z, np.minimum(quads_pos, obst_max_pos))
            obst_z = self.pos[2]
            quads_z = pos_quads[:, 2]
            delta_z = abs(np.maximum(obst_z - obst_half_height, np.minimum(quads_z, obst_z + obst_half_height)) - quads_z)

            rel_closest_dist = np.array([delta_xy_dist, delta_z]).T
            dist_arr = np.linalg.norm(rel_closest_dist, axis=1)

        collision_arr = (dist_arr <= self.quad_size).astype(np.float32)
        return collision_arr, dist_arr

    def collision_detection(self, pos_quads=None):
        if self.shape == 'cube':
            collision_arr, dist_arr = self.cube_detection(pos_quads)
        elif self.shape == 'sphere':
            collision_arr, dist_arr = self.sphere_detection(pos_quads)
        elif self.shape == 'cylinder':
            collision_arr, dist_arr = self.cylinder_detection(pos_quads)
        else:
            raise NotImplementedError()

        return collision_arr, dist_arr

    @staticmethod
    def get_dir_of_cube_collision(drone_dyn, obstacle_dyn, obst_z_size):
        drone_pos = drone_dyn.pos
        obst_pos = obstacle_dyn.pos
        shift_pos = drone_pos - obst_pos
        abs_shift_pos = np.abs(shift_pos)
        x_list = [abs_shift_pos[0] >= abs_shift_pos[1] and shift_pos[0] >= 0,
                  abs_shift_pos[0] >= abs_shift_pos[1] and shift_pos[0] < 0]
        y_list = [abs_shift_pos[0] < abs_shift_pos[1] and shift_pos[1] >= 0,
                  abs_shift_pos[0] < abs_shift_pos[1] and shift_pos[1] < 0]
        z_list = [abs_shift_pos[2] > obst_z_size and shift_pos[2] >= 0,
                  abs_shift_pos[2] > obst_z_size and shift_pos[2] < 0]

        direction = np.random.uniform(low=-1.0, high=1.0, size=(3,))
        if drone_pos[2] == 0.0:
            direction[2] = np.random.uniform(low=-0.01, high=0.01)

        if x_list[0]:
            direction[0] = np.random.uniform(low=0.1, high=1.0)
        elif x_list[1]:
            direction[0] = np.random.uniform(low=-1.0, high=-0.1)

        if y_list[0]:
            direction[1] = np.random.uniform(low=0.1, high=1.0)
        elif y_list[1]:
            direction[1] = np.random.uniform(low=-1.0, high=-0.1)

        # if any item in z_list, we should reconsider direction in xy plane
        if z_list[0]:
            direction[:2] = np.random.uniform(low=-1.0, high=1.0, size=(2,))
            direction[2] = np.random.uniform(low=0.1, high=1.0)
        elif z_list[1]:
            direction[:2] = np.random.uniform(low=-1.0, high=1.0, size=(2,))
            direction[2] = np.random.uniform(low=-1.0, high=-0.5)

        return direction

    @staticmethod
    def get_dir_of_cylinder_collision(drone_dyn, obstacle_dyn, obst_z_size):
        drone_pos = drone_dyn.pos
        obst_pos = obstacle_dyn.pos
        shift_pos = drone_pos - obst_pos
        abs_shift_pos = np.abs(shift_pos)
        sx, sy, _ = shift_pos

        z_list = [abs_shift_pos[2] > obst_z_size and shift_pos[2] >= 0,
                  abs_shift_pos[2] > obst_z_size and shift_pos[2] < 0]

        direction = shift_pos
        direction_mag = np.linalg.norm(direction)
        direction = direction / (direction_mag + 0.00001 if direction_mag == 0.0 else direction_mag)

        direction += np.random.uniform(low=-0.2 * direction, high=0.2*direction)

        if drone_pos[2] == 0.0:
            direction[2] = np.random.uniform(low=-0.01, high=0.01)

        # if any item in z_list, we should reconsider direction in xy plane
        if z_list[0]:
            direction[:2] = np.random.uniform(low=-1.0, high=1.0, size=(2,))
            direction[2] = np.random.uniform(low=0.1, high=1.0)
        elif z_list[1]:
            direction[:2] = np.random.uniform(low=-1.0, high=1.0, size=(2,))
            direction[2] = np.random.uniform(low=-1.0, high=-0.5)

        return direction

    def get_dir_of_collision(self, drone_dyn, obstacle_dyn, obst_z_size):
        if self.shape == 'cube':
            direction = self.get_dir_of_cube_collision(drone_dyn=drone_dyn, obstacle_dyn=obstacle_dyn,
                                                       obst_z_size=obst_z_size)
        elif self.shape == 'cylinder':
            direction = self.get_dir_of_cylinder_collision(drone_dyn=drone_dyn, obstacle_dyn=obstacle_dyn,
                                                           obst_z_size=obst_z_size)
        else:
            raise NotImplementedError()

        return direction

    def inside_collision_detection(self, pos_quad=None):
        if self.inf_height:
            obst_z_size = 0.5 * self.room_dims[2]
        else:
            obst_z_size = 0.5 * self.size

        rel_pos = pos_quad - self.pos
        obst_range = 0.5 * self.size + 0.001

        if self.shape == 'cube':
            if abs(rel_pos[0]) <= obst_range and abs(rel_pos[1]) <= obst_range and abs(rel_pos[2]) <= obst_z_size:
                return True
            else:
                return False
        elif self.shape == 'sphere':
            if np.linalg.norm(rel_pos) <= obst_range:
                return True
            else:
                return False
        elif self.shape == 'cylinder':
            if np.linalg.norm(rel_pos[:2]) <= obst_range and abs(rel_pos[2]) <= obst_z_size:
                return True
            else:
                return False
        else:
            raise NotImplementedError(f'inside_collision_detection, {self.shape} not supported!')

    def get_closest_points(self, quads_pos):
        if self.shape == 'cube':
            if self.inf_height:
                obst_min_pos = self.pos - np.array([0.5 * self.size, 0.5 * self.size, 0.5 * self.room_dims[2]])
                obst_max_pos = self.pos + np.array([0.5 * self.size, 0.5 * self.size, 0.5 * self.room_dims[2]])
            else:
                obst_min_pos = self.pos - 0.5 * self.size
                obst_max_pos = self.pos + 0.5 * self.size

            closest_points = np.maximum(obst_min_pos, np.minimum(quads_pos, obst_max_pos))

        elif self.shape == 'cylinder':
            if self.inf_height:
                rel_pos_xy = self.pos[:2] - quads_pos[:, :2]
                rel_dist_xy = np.linalg.norm(rel_pos_xy, axis=1)
                rel_dist_xy = np.maximum(rel_dist_xy, 1e-6)
                rel_pos_xy_unit = rel_pos_xy / rel_dist_xy[:, None]
                closest_points = rel_pos_xy - rel_pos_xy_unit * (0.5 * self.size)
            else:
                raise NotImplementedError(f'quadrotor_single_obstacle: Not inf height, not supported!')

        else:
            raise NotImplementedError(f'self.shape: {self.shape} not supported!')



        return closest_points
