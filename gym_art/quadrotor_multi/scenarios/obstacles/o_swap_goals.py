import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base


class Scenario_o_swap_goals(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # teleport every [4.0, 6.0] secs
        duration_time = 6.0
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

    def update_goals(self):
        np.random.shuffle(self.goals)
        for env, goal in zip(self.envs, self.goals):
            env.goal = goal

    def step(self):
        tick = self.envs[0].tick
        # Switch every [4, 6] seconds
        if tick % self.control_step_for_sec == 0 and tick > 0:
            self.update_goals()

        return

    def reset(self, params):
        self.obstacle_map = params['obst_map']
        self.cell_centers = params['cell_centers']
        self.sim2real_scenario = params.get('sim2real_scenario', None)

        if self.obstacle_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        # Update duration time
        duration_time = np.random.uniform(low=4.0, high=6.0)
        self.control_step_for_sec = int(duration_time * self.envs[0].control_freq)

        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        self.start_point = self.generate_pos_obst_map_2(num_agents=self.num_agents)
        self.spawn_points = copy.deepcopy(self.start_point)

        self.formation_center = self.max_square_area_center()

        # Regenerate goals, we don't have to assign goals to the envs,
        # the reset function in quadrotor_multi.py would do that
        self.goals = self.generate_goals(num_agents=self.num_agents, formation_center=self.formation_center,
                                         layer_dist=self.layer_dist)
        np.random.shuffle(self.goals)
