import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base
from gym_art.quadrotor_multi.quadrotor_traj_gen import QuadTrajGen
from gym_art.quadrotor_multi.quadrotor_planner import traj_eval


class Scenario_o_static_same_goal(Scenario_o_base):
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        """ This scenario implements a 13 dim goal that tracks a smooth polynomial trajectory. For a static goal, we
            use only the last time point as a goal. This would mean all derivatives and higher order derivatives would 
            be zero. """
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5
        self.goal_generator = []

    def step(self):
        # Goal point does not change, so we just pass by step.
        return

    def reset(self, obst_map=None, cell_centers=None):
        
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        if obst_map is None or cell_centers is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))

        self.start_point = []
        self.end_point = []

        for i in range(self.num_agents):
            self.start_point.append(self.generate_pos_obst_map())
            
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[i])
            
            self.goal_generator.append(QuadTrajGen(poly_degree=7))
            
            final_goal = self.generate_pos_obst_map()
            
            # Fix the goal height at 0.65 m
            final_goal[2] = 0.65
            # Generate trajectory with random time from (2, ep_time)
            self.goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=np.append(final_goal, np.random.uniform(low=0, high=3.14)), 
                                                   duration=np.random.uniform(low=2, high=self.envs[0].ep_time), current_time=0)
            
            #Find the initial goal
            self.end_point.append(self.goal_generator[i].piecewise_eval(self.envs[0].ep_time).as_nparray())

        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        
        # Reset formation and related parameters
        self.update_formation_and_relate_param()

        # Reassign goals
        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = copy.deepcopy(self.end_point)
