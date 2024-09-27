import numpy as np

from gym_art.quadrotor_multi.scenarios.base import QuadrotorScenario
from gym_art.quadrotor_multi.scenarios.utils import QUADS_MODE_LIST_SINGLE, QUADS_MODE_LIST, \
    QUADS_MODE_LIST_OBSTACLES, QUADS_MODE_LIST_OBSTACLES_SINGLE

# Neighbor Scenarios
from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.dynamic_diff_goal import Scenario_dynamic_diff_goal
from gym_art.quadrotor_multi.scenarios.dynamic_formations import Scenario_dynamic_formations
from gym_art.quadrotor_multi.scenarios.dynamic_same_goal import Scenario_dynamic_same_goal
from gym_art.quadrotor_multi.scenarios.ep_lissajous3D import Scenario_ep_lissajous3D
from gym_art.quadrotor_multi.scenarios.ep_rand_bezier import Scenario_ep_rand_bezier
from gym_art.quadrotor_multi.scenarios.run_away import Scenario_run_away
from gym_art.quadrotor_multi.scenarios.static_diff_goal import Scenario_static_diff_goal
from gym_art.quadrotor_multi.scenarios.static_same_goal import Scenario_static_same_goal
from gym_art.quadrotor_multi.scenarios.swap_goals import Scenario_swap_goals
from gym_art.quadrotor_multi.scenarios.swarm_vs_swarm import Scenario_swarm_vs_swarm

# Obstacles
from gym_art.quadrotor_multi.scenarios.obstacles.o_random import Scenario_o_random
from gym_art.quadrotor_multi.scenarios.obstacles.o_static_same_goal import Scenario_o_static_same_goal
from gym_art.quadrotor_multi.scenarios.obstacles.o_swap_goals import Scenario_o_swap_goals
from gym_art.quadrotor_multi.scenarios.obstacles.o_ep_rand_bezier import Scenario_o_ep_rand_bezier
from gym_art.quadrotor_multi.scenarios.obstacles.o_random_dynamic_goal import Scenario_o_random_dynamic_goal
from gym_art.quadrotor_multi.scenarios.obstacles.o_random_dynamic_goal_curriculum import Scenario_o_random_dynamic_goal_curriculum
# Test Scenarios
from gym_art.quadrotor_multi.scenarios.test.o_test import Scenario_o_test


def create_scenario(quads_mode, envs, num_agents, room_dims):
    cls = eval('Scenario_' + quads_mode)
    scenario = cls(quads_mode, envs, num_agents, room_dims)
    return scenario


class Scenario_mix(QuadrotorScenario):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)

        # Once change the parameter here, should also update QUADS_PARAMS_DICT to make sure it is same as run a
        # single scenario key: quads_mode value: 0. formation, 1: [formation_low_size, formation_high_size],
        # 2: episode_time
        if num_agents == 1:
            if envs[0].use_obstacles:
                self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES_SINGLE
            else:
                self.quads_mode_list = QUADS_MODE_LIST_SINGLE
        elif num_agents > 1 and not envs[0].use_obstacles:
            self.quads_mode_list = QUADS_MODE_LIST
        elif envs[0].use_obstacles:
            self.quads_mode_list = QUADS_MODE_LIST_OBSTACLES
        else:
            raise NotImplementedError("Unknown!")

        self.scenario = None
        self.approch_goal_metric = 0.5

        self.spawn_points = None

    def name(self):
        """
        :return: the name of the actual scenario used in this episode
        """
        return self.scenario.__class__.__name__

    def step(self):
        self.scenario.step()

        # We change goals dynamically
        self.goals = self.scenario.goals

        # Rendering
        self.formation_size = self.scenario.formation_size
        return

    def reset(self, obst_map=None, cell_centers=None):
        mode_index = np.random.randint(low=0, high=len(self.quads_mode_list))
        mode = self.quads_mode_list[mode_index]

        # Init the scenario
        self.scenario = create_scenario(quads_mode=mode, envs=self.envs, num_agents=self.num_agents,
                                        room_dims=self.room_dims)

        if obst_map is not None:
            self.scenario.reset(obst_map, cell_centers)
        else:
            self.scenario.reset()

        self.goals = self.scenario.goals
        self.spawn_points = self.scenario.spawn_points
        self.formation_size = self.scenario.formation_size
        self.approch_goal_metric = self.scenario.approch_goal_metric
