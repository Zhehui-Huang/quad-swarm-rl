import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from gym_art.quadrotor_multi.scenarios.obstacles.o_utils import get_goals_given_formation
from gym_art.quadrotor_multi.scenarios.utils import get_goal_by_formation


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
        self.obst_spawn_area = envs[0].obst_spawn_area

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

    def reset(self, obst_map, cell_centers):
        self.start_point = self.generate_pos()
        self.end_point = self.generate_pos()
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.standard_reset(formation_center=self.start_point)

    def generate_pos_obst_map(self, check_surroundings=False):
        idx = np.random.choice(a=len(self.free_space), replace=False)
        x, y = self.free_space[idx][0], self.free_space[idx][1]
        if check_surroundings:
            surroundings_free = self.check_surroundings(x, y)
            while not surroundings_free:
                idx = np.random.choice(a=len(self.free_space), replace=False)
                x, y = self.free_space[idx][0], self.free_space[idx][1]
                surroundings_free = self.check_surroundings(x, y)

        width = self.obstacle_map.shape[0]
        index = x + (width * y)
        pos_x, pos_y = self.cell_centers[index]
        z_list_start = np.random.uniform(low=0.5, high=1.0)
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
            z_list_start = np.random.uniform(low=0.5, high=1.0)
            generated_points.append(np.array([pos_x, pos_y, z_list_start]))

        return np.array(generated_points)

    def generate_start_goal_pos(self, pos_area_flag, goal_scenario_flag, formation, num_agents):
        pos_shift = 1.0
        step_size = 0.5

        room_width, room_depth = self.room_dims[0], self.room_dims[1]
        obst_area_width, obst_area_depth = self.obst_spawn_area[0], self.obst_spawn_area[1]

        pos_x_min = -room_width / 2 + pos_shift
        pos_x_max = room_width / 2 - pos_shift
        pos_x_grids = np.arange(pos_x_min, pos_x_max + step_size, step_size)

        pos_y1_min = -room_depth / 2 + pos_shift
        pos_y1_max = -obst_area_depth / 2 - pos_shift

        pos_y2_min = obst_area_depth / 2 + pos_shift
        pos_y2_max = room_depth / 2 - pos_shift

        if pos_area_flag == 0:
            pos_y_grids = np.arange(pos_y1_min, pos_y1_max + 0.1, step_size)
            goal_y_grids = np.arange(pos_y2_min, pos_y2_max + 0.1, step_size)
        else:
            pos_y_grids = np.arange(pos_y2_min, pos_y2_max + 0.1, step_size)
            goal_y_grids = np.arange(pos_y1_min, pos_y1_max + 0.1, step_size)

        start_pos = []
        noise_size = 0.1

        all_pairs = np.array(np.meshgrid(pos_x_grids, pos_y_grids)).T.reshape(-1, 2)
        selected_pairs = all_pairs[np.random.choice(all_pairs.shape[0], num_agents, replace=False)]
        noise = np.random.uniform(low=-noise_size, high=noise_size, size=selected_pairs.shape)
        noisy_selected_pairs = selected_pairs + noise
        z_list_start = np.random.uniform(low=0.5, high=1.0, size=num_agents)

        for i in range(num_agents):
            pos_x, pos_y = noisy_selected_pairs[i]
            pos_item = np.array([pos_x, pos_y, z_list_start[i]])
            start_pos.append(pos_item)

        start_pos = np.array(start_pos)

        if goal_scenario_flag:
            # same goal
            pos_x = np.random.choice(pos_x_grids)
            pos_x = np.clip(a=pos_x, a_min=-room_width / 2 + pos_shift, a_max=room_width / 2 - pos_shift)
            pos_x += np.random.uniform(low=-0.2, high=0.2)

            pos_y = np.random.choice(goal_y_grids)
            pos_y = np.clip(a=pos_y, a_min=-room_depth / 2 + pos_shift, a_max=room_depth / 2 - pos_shift)
            pos_y += np.random.uniform(low=-0.2, high=0.2)

            formation_center = np.array([pos_x, pos_y, 0.65])
            goal_pos_list = get_goals_given_formation(
                formation=formation, dist_range=[0.0, 0.0], formation_center=formation_center, num_agents=num_agents
            )
        else:
            goal_all_pairs = np.array(np.meshgrid(pos_x_grids, goal_y_grids)).T.reshape(-1, 2)
            goal_selected_pairs = goal_all_pairs[np.random.choice(goal_all_pairs.shape[0], num_agents, replace=False)]
            goal_noise = np.random.uniform(low=-noise_size, high=noise_size, size=selected_pairs.shape)
            goal_pos_list = goal_selected_pairs + goal_noise

        goal_pos = []
        for goal_item in goal_pos_list:
            x, y = goal_item[0], goal_item[1]
            x = np.clip(a=x, a_min=-room_width / 2 + 0.5, a_max=room_width / 2 - 0.5)
            y = np.clip(a=y, a_min=-room_depth / 2 + 0.5, a_max=room_depth / 2 - 0.5)
            goal_pos.append(np.array([x, y, 0.65]))

        return np.array(start_pos), np.array(goal_pos)

    def check_surroundings(self, row, col):
        length, width = self.obstacle_map.shape[0], self.obstacle_map.shape[1]
        obstacle_map = self.obstacle_map
        # Check if the given position is out of bounds
        if row < 0 or row >= width or col < 0 or col >= length:
            raise ValueError("Invalid position")

        # Check if the surrounding cells are all 0s
        check_pos_x, check_pos_y = [], []
        if row > 0:
            check_pos_x.append(row - 1)
            check_pos_y.append(col)
            if col > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col - 1)
            if col < length - 1:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
        if row < width - 1:
            check_pos_x.append(row + 1)
            check_pos_y.append(col)

        if col > 0:
            check_pos_x.append(row)
            check_pos_y.append(col - 1)
        if col < length - 1:
            check_pos_x.append(row)
            check_pos_y.append(col + 1)
            if row > 0:
                check_pos_x.append(row - 1)
                check_pos_y.append(col + 1)
            if row < length - 1:
                check_pos_x.append(row + 1)
                check_pos_y.append(col + 1)

        check_pos = ([check_pos_x, check_pos_y])
        # Get the values of the adjacent cells
        adjacent_cells = obstacle_map[tuple(check_pos)]

        return np.any(adjacent_cells != 0)

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
        z_list_start = np.random.uniform(low=0.5, high=1.0)
        return np.array([pos_x, pos_y, z_list_start])
