import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_random(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5

    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        # tick = self.envs[0].tick
        #
        # if tick <= self.duration_step:
        #     return

        # self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        # for i, env in enumerate(self.envs):
        #     env.goal = self.end_point[i]

        return

    def reset(self, obst_map, cell_centers, sim2real_scenario=None):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None:
            raise NotImplementedError

        if sim2real_scenario is None:
            obst_map_locs = np.where(self.obstacle_map == 0)
            self.free_space = list(zip(*obst_map_locs))

            self.start_point = []
            self.end_point = []
            for i in range(self.num_agents):
                self.start_point.append(self.generate_pos_obst_map())
                self.end_point.append(self.generate_pos_obst_map())

            self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
            self.end_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
            for i in range(len(self.end_point)):
                self.end_point[i][2] = 0.65
        else:
            self.start_point = [[-1.25, 0.25, 0.65]]
            self.end_point = [[2.75, 0.25, 0.65]]
        # self.start_point = self.generate_pos_obst_map_2(self.num_agents)
        # self.end_point = self.generate_pos_obst_map_2(self.num_agents)

        # self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = copy.deepcopy(self.end_point)
