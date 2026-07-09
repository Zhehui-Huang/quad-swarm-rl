import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario


class Scenario_o_base(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.start_point = np.array([0.0, -3.0, 2.0])
        self.end_point = np.array([0.0, 3.0, 2.0])
        self.room_dims = room_dims
        self.duration_step = 0
        self.quads_mode = quads_mode
        self.obstacle_map = None
        self.free_space = []
        self.approch_goal_metric = 1.0

        self.spawn_points = None
        self.cell_centers = None

    def generate_pos(self):
        half_room_length = self.room_dims[0] / 2
        half_room_width = self.room_dims[1] / 2

        x = np.random.uniform(low=-1.0 * half_room_length + 2.0, high=half_room_length - 2.0)
        y = np.random.uniform(low=-1.0 * half_room_width + 2.0, high=half_room_width - 2.0)

        z = np.random.uniform(low=1.0, high=4.0)

        return np.array([x, y, z])

    def step(self):
        tick = self.envs[0].tick

        if tick <= self.duration_step:
            return

        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.end_point, layer_dist=0.0)

        for i, env in enumerate(self.envs):
            env.goal = self.goals[i]

        return

    def reset(self, obst_map=None, cell_centers=None):
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.standard_reset(formation_center=self.start_point)

    def generate_pos_obst_map(self, check_surroundings=False):
        idx = np.random.choice(a=len(self.free_space), replace=False)
        x, y = self.free_space[idx][0], self.free_space[idx][1]
        if check_surroundings:
            surroundings_free = self.check_surroundings(x, y)
            retries = 0
            while not surroundings_free and retries < 100:
                idx = np.random.choice(a=len(self.free_space), replace=False)
                x, y = self.free_space[idx][0], self.free_space[idx][1]
                surroundings_free = self.check_surroundings(x, y)
                retries += 1

        width = self.obstacle_map.shape[0]
        index = x + (width * y)
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=0.75, high=3.0)
        # xy_noise = np.random.uniform(low=-0.2, high=0.2, size=2)
        return np.array([pos_x, pos_y, z_list_start])

    def generate_pos_obst_map_2(self, num_agents):
        ids = np.random.choice(range(len(self.free_space)), num_agents, replace=False)

        generated_points = []
        for idx in ids:
            x, y = self.free_space[idx][0], self.free_space[idx][1]
            width = self.obstacle_map.shape[0]
            index = x + (width * y)
            pos_x, pos_y = self.cell_centers[index]
            z_list_start = np.random.uniform(low=1.0, high=3.0)
            generated_points.append(np.array([pos_x, pos_y, z_list_start]))

        return np.array(generated_points)

    def generate_pos_obst_map_side(self, side='left', z_low=1.0, z_high=3.0):
        width = self.obstacle_map.shape[0]
        free_positions = []
        for idx in range(len(self.free_space)):
            x, y = self.free_space[idx]
            index = x + (width * y)
            pos_x, pos_y = self.cell_centers[index]
            free_positions.append((pos_x, pos_y))

        if len(free_positions) == 0:
            raise ValueError('No free space available in obstacle map')

        xs = np.array([p[0] for p in free_positions])
        median_x = np.median(xs)
        if side == 'left':
            candidates = [p for p in free_positions if p[0] <= median_x]
        elif side == 'right':
            candidates = [p for p in free_positions if p[0] >= median_x]
        elif side == 'center':
            center_width = self.room_dims[0] * 0.25
            lower = median_x - center_width
            upper = median_x + center_width
            candidates = [p for p in free_positions if lower <= p[0] <= upper]
        else:
            raise ValueError(f"Unknown side '{side}' for obstacle position generation")
        if len(candidates) == 0:
            candidates = free_positions

        pos_x, pos_y = candidates[np.random.choice(len(candidates))]
        z_list_start = np.random.uniform(low=z_low, high=z_high)
        return np.array([pos_x, pos_y, z_list_start])

    def generate_pos_obst_map_side_batch(self, side='left', num_agents=1):
        generated_points = []
        for _ in range(num_agents):
            generated_points.append(self.generate_pos_obst_map_side(side))
        return np.array(generated_points)

    def check_surroundings(self, row, col):
        length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
        obstacle_map = self.obstacle_map
        # Check if the given position is out of bounds
        if row < 0 or row >= length or col < 0 or col >= width:
            raise ValueError("Invalid position")

        # Check if the surrounding cells are all 0s
        check_pos_x, check_pos_y = [], []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                r_new, c_new = row + dr, col + dc
                if 0 <= r_new < length and 0 <= c_new < width:
                    check_pos_x.append(r_new)
                    check_pos_y.append(c_new)

        check_pos = (check_pos_x, check_pos_y)
        # Get the values of the adjacent cells
        adjacent_cells = obstacle_map[tuple(check_pos)]

        return not np.any(adjacent_cells != 0)

    def max_square_area_center(self):
        """
        Finds the maximum square area of 0 in a 2D matrix and returns the coordinates
        of the center element of the largest square area.
        """
        n, m = self.obstacle_map.shape
        # Initialize a 2D numpy array to store the maximum size of square submatrices
        # that end at each element of the matrix.
        dp = np.zeros((n, m), dtype=int)
        # Initialize the first row and first column of the dp array
        dp[0] = self.obstacle_map[0]
        dp[:, 0] = self.obstacle_map[:, 0]
        # Initialize variables to store the maximum square area and its center coordinates
        max_size = 0
        center_x = 0
        center_y = 0
        # Fill the remaining entries of the dp array
        for i in range(1, n):
            for j in range(1, m):
                if self.obstacle_map[i][j] == 0:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
                    if dp[i][j] > max_size:
                        max_size = dp[i][j]
                        center_x = i - (max_size - 1) // 2
                        center_y = j - (max_size - 1) // 2
        # Return the center coordinates of the largest square area as a tuple
        index = center_x + (m * center_y)
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=1.5, high=3.0)
        return np.array([pos_x, pos_y, z_list_start])
