import numpy as np
import copy
import bezier

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_ep_rand_bezier(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        duration_time = 0.3
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)
        self.approch_goal_metric = 1.0

    def step(self):
        # randomly sample new goal pos in free space and have the goal move there following a bezier curve
        tick = self.envs[0].tick
        control_freq = self.envs[0].control_freq
        num_secs = 6
        control_steps = int(num_secs * control_freq)
        t = tick % control_steps
        room_dims = np.array(self.room_dims) - self.formation_size
        # min and max distance the goal can spawn away from its current location. 30 = empirical upper bound on
        # velocity that the drones can handle.
        max_dist = min(5, max(room_dims))
        min_dist = max_dist / 2
        if tick % control_steps == 0 or tick == 1:
            # sample a new goal pos that's within the room boundaries and satisfies the distance constraint
            new_goal_found = False
            while not new_goal_found:
                low, high = np.array([-room_dims[0] / 2, -room_dims[1] / 2, 1.5]), np.array(
                    [room_dims[0] / 2, room_dims[1] / 2, 3.0])
                # need an intermediate point for a deg=2 curve
                new_pos = np.random.uniform(low=-high, high=high, size=(2, 3)).reshape(3, 2)
                # add some velocity randomization = random magnitude * unit direction
                new_pos = new_pos * np.random.randint(min_dist, max_dist + 1) / np.linalg.norm(new_pos, axis=0)
                new_pos = self.goals[0].reshape(3, 1) + new_pos
                lower_bound = np.expand_dims(low, axis=1)
                upper_bound = np.expand_dims(high, axis=1)
                new_goal_found = (new_pos > lower_bound + 0.5).all() and (
                            new_pos < upper_bound - 0.5).all()  # check bounds that are slightly smaller than the room dims
            # new_pos = np.append(self.sampled_points[t], 2.0)
            # new_pos = new_pos.reshape(3, 1)
            nodes = np.concatenate((self.goals[0].reshape(3, 1), new_pos), axis=1)
            nodes = np.asfortranarray(nodes)
            pts = np.linspace(0, 1, control_steps)
            curve = bezier.Curve(nodes, degree=2)
            self.interp = curve.evaluate_multi(pts)
            # self.interp = np.clip(self.interp, a_min=np.array([0,0,0.2]).reshape(3,1), a_max=high.reshape(3,
            # 1)) # want goal clipping to be slightly above the floor
        if tick % control_steps != 0 and tick > 1:
            self.goals = np.array([self.interp[:, t] for _ in range(self.num_agents)])

            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

        return

    def reset(self, params):
        # Update duration time
        self.duration_time = 0.01
        self.control_step_for_sec = int(self.duration_time * self.envs[0].control_freq)

        self.obstacle_map = params['obst_map']
        self.cell_centers = params['cell_centers']
        if self.obstacle_map is None or self.cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.end_point = self.generate_pos_obst_map()

        # Generate obstacle-free trajectory points
        num_samples = 10
        max_dist = 4.0
        sampled_points_idx = []
        while len(sampled_points_idx) < num_samples:
            # Randomly select a point
            point_idx = np.random.choice(len(self.free_space))

            # Check if the distance constraint is satisfied with the previously sampled points
            if len(sampled_points_idx) > 0:
                distances = np.array([np.linalg.norm(self.cell_centers[sampled_point_idx] - self.cell_centers[point_idx])
                                      for sampled_point_idx in sampled_points_idx])
                if np.any(distances > max_dist):
                    continue

            # Add the point to the sampled trajectory and remove it from the free space
            sampled_points_idx.append(point_idx)
            self.free_space.pop(point_idx)

        # Separate x and y coordinates of the sampled points
        self.sampled_points = self.cell_centers[sampled_points_idx]

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reassign goals
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = np.array([self.end_point for _ in range(self.num_agents)])
