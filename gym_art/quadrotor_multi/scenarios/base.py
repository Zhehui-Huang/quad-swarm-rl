import numpy as np

from gym_art.quadrotor_multi.scenarios.utils import QUADS_PARAMS_DICT, update_formation_and_max_agent_per_layer, \
    update_layer_dist, get_formation_range, get_goal_by_formation
from gym_art.quadrotor_multi.scenarios.utils import generate_points, get_grid_dim_number


class QuadrotorScenario:
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        self.quads_mode = quads_mode
        self.envs = envs
        self.num_agents = num_agents
        self.room_dims = room_dims
        self.goals = None

        #  Set formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size,
        #  layer_dist, formation_center
        #  Note: num_agents_per_layer for scalability, the maximum number of agent per layer
        self.formation = None
        self.formation_center = None
        self.lowest_formation_size, self.highest_formation_size = 1.0, 2.0
        self.formation_size = 1.0

        self.num_agents_per_layer = 8
        self.layer_dist = self.lowest_formation_size

        # Aux variables for scenario: pursuit evasion
        self.interp = None
        # Aux variables used in scenarios with obstacles
        self.spawn_points = None
        self.approch_goal_metric = 0.5

    def name(self):
        """
        :return: scenario name
        """
        return self.__class__.__name__

    def generate_goals(self, num_agents, formation_center=None, layer_dist=0.0):
        if formation_center is None:
            formation_center = np.array([0., 0., 2.])

        if self.formation.startswith("circle"):
            if num_agents <= self.num_agents_per_layer:
                real_num_per_layer = [num_agents]
            else:
                whole_layer_num = num_agents // self.num_agents_per_layer
                real_num_per_layer = [self.num_agents_per_layer for _ in range(whole_layer_num)]
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    real_num_per_layer.append(rest_num)

            pi = np.pi
            goals = []
            for i in range(num_agents):
                cur_layer_num_agents = real_num_per_layer[i // self.num_agents_per_layer]
                degree = 2 * pi * (i % cur_layer_num_agents) / cur_layer_num_agents
                pos_0 = self.formation_size * np.cos(degree)
                pos_1 = self.formation_size * np.sin(degree)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1,
                                             layer_pos=(i // self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            goals = np.array(goals)
            goals += formation_center

        elif self.formation == "sphere":
            goals = self.formation_size * np.array(generate_points(num_agents)) + formation_center

        elif self.formation.startswith("grid"):
            if num_agents <= self.num_agents_per_layer:
                dim_1, dim_2 = get_grid_dim_number(num_agents)
                dim_size_each_layer = [[dim_1, dim_2]]
            else:
                # whole layer
                whole_layer_num = num_agents // self.num_agents_per_layer
                max_dim_1, max_dim_2 = get_grid_dim_number(self.num_agents_per_layer)
                dim_size_each_layer = [[max_dim_1, max_dim_2] for _ in range(whole_layer_num)]

                # deal with the rest of the drones
                rest_num = num_agents % self.num_agents_per_layer
                if rest_num > 0:
                    dim_1, dim_2 = get_grid_dim_number(rest_num)
                    dim_size_each_layer.append([dim_1, dim_2])

            goals = []
            for i in range(num_agents):
                dim_1, dim_2 = dim_size_each_layer[i // self.num_agents_per_layer]
                pos_0 = self.formation_size * (i % dim_2)
                pos_1 = self.formation_size * (int(i / dim_2) % dim_1)
                goal = get_goal_by_formation(formation=self.formation, pos_0=pos_0, pos_1=pos_1,
                                             layer_pos=(i // self.num_agents_per_layer) * layer_dist)
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        elif self.formation.startswith("cube"):
            dim_size = np.power(num_agents, 1.0 / 3)
            floor_dim_size = int(dim_size)
            goals = []
            for i in range(num_agents):
                pos_0 = self.formation_size * (int(i / floor_dim_size) % floor_dim_size)
                pos_1 = self.formation_size * (i % floor_dim_size)
                goal = np.array(
                    [formation_center[2] + self.formation_size * (i // np.square(floor_dim_size)), pos_0, pos_1])
                goals.append(goal)

            mean_pos = np.mean(goals, axis=0)
            goals = goals - mean_pos + formation_center
        else:
            raise NotImplementedError("Unknown formation")

        return goals

    def update_formation_size(self, new_formation_size):
        if new_formation_size != self.formation_size:
            self.formation_size = new_formation_size if new_formation_size > 0.0 else 0.0
            self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                             layer_dist=self.layer_dist)
            for i, env in enumerate(self.envs):
                env.goal = self.goals[i]

    def update_formation_and_relate_param(self):
        # Reset formation, num_agents_per_layer, lowest_formation_size, highest_formation_size, formation_size,
        # layer_dist
        self.formation, self.num_agents_per_layer = update_formation_and_max_agent_per_layer(mode=self.quads_mode)
        # QUADS_PARAMS_DICT:
        # Key: quads_mode; Value: 0. formation, 1: [formation_low_size, formation_high_size], 2: episode_time
        lowest_dist, highest_dist = QUADS_PARAMS_DICT[self.quads_mode][1]
        self.lowest_formation_size, self.highest_formation_size = \
            get_formation_range(mode=self.quads_mode, formation=self.formation, num_agents=self.num_agents,
                                low=lowest_dist, high=highest_dist, num_agents_per_layer=self.num_agents_per_layer)

        self.formation_size = np.random.uniform(low=self.lowest_formation_size, high=self.highest_formation_size)
        self.layer_dist = update_layer_dist(low=self.lowest_formation_size, high=self.highest_formation_size)

    def step(self):
        raise NotImplementedError("Implemented in a specific scenario")

    def reset(self, params):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset formation center
        self.formation_center = np.array([0.0, 0.0, 2.0])

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)

    def standard_reset(self, formation_center=None):
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reset formation center
        if formation_center is None:
            self.formation_center = np.array([0.0, 0.0, 2.0])
        else:
            self.formation_center = formation_center

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)
