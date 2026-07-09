import numpy as np
import copy
from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base

class Scenario_o_random(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5
        self.waypoint_threshold = 0.7
        self.goal_success_first = 10.0
        self.goal_success_later = 1.0
        self.first_goal_completed = False
        self.goal_z = self.room_dims[2] * 0.5

    def update_formation_size(self, new_formation_size):
        pass

    def generate_new_goal_for_agent(self, agent_idx):
        current_pos = self.envs[agent_idx].dynamics.pos

        retries = 0
        while True:
            retries += 1
            if self.obstacle_map is not None:
                candidate = self.generate_pos_obst_map(check_surroundings=True)
                candidate[2] = self.goal_z
            else:
                half_room_length = self.room_dims[0] / 2
                half_room_width = self.room_dims[1] / 2
                x = np.random.uniform(low=-half_room_length + 1.0, high=half_room_length - 1.0)
                y = np.random.uniform(low=-half_room_width + 1.0, high=half_room_width - 1.0)
                candidate = np.array([x, y, self.goal_z])

            if np.linalg.norm(candidate - current_pos) > 3.0:
                return candidate

            # Ultimate fallback
            if retries > 200:
                return candidate

    def generate_new_goals(self):
        return np.array([self.generate_new_goal_for_agent(i) for i in range(self.num_agents)])

    def set_goal_success_reward(self, success_reward):
        for env in self.envs:
            env.rew_coeff["success"] = success_reward

    def step(self):
        goal_changed = False
        for i, env in enumerate(self.envs):
            if np.linalg.norm(env.dynamics.pos - self.goals[i]) < self.waypoint_threshold:
                if not self.first_goal_completed:
                    self.first_goal_completed = True
                    self.set_goal_success_reward(self.goal_success_later)
                self.goals[i] = self.generate_new_goal_for_agent(i)
                goal_changed = True
            env.goal = self.goals[i]

        return goal_changed


    def reset(self, obst_map=None, cell_centers=None):
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        
        if obst_map is not None:
            obst_map_locs = np.where(self.obstacle_map == 0)
            self.free_space = list(zip(*obst_map_locs))
            self.start_point = self.generate_pos_obst_map_side_batch(side='left', num_agents=self.num_agents)
        else:
            self.start_point = []
            half_room_length = self.room_dims[0] / 2
            half_room_width = self.room_dims[1] / 2
            for _ in range(self.num_agents):
                x = np.random.uniform(low=-half_room_length + 1.0, high=-half_room_length / 3.0)
                y = np.random.uniform(low=-half_room_width + 1.0, high=half_room_width - 1.0)
                self.start_point.append([x, y, self.goal_z])
            self.start_point = np.array(self.start_point)

        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.update_formation_and_relate_param()

        self.first_goal_completed = False

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        
        # Initialize goals
        self.goals = np.zeros((self.num_agents, 3))
        for i in range(self.num_agents):
            self.envs[i].dynamics.pos = self.spawn_points[i] # Just to temporally set for distance calculation
        self.goals = self.generate_new_goals()
