
import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base
from gym_art.quadrotor_multi.quadrotor_traj_gen import QuadTrajGen
from gym_art.quadrotor_multi.quadrotor_planner import traj_eval


class Scenario_o_random_dynamic_goal_curriculum(Scenario_o_base):
    """ This scenario implements a 13 dim goal that tracks a smooth polynomial trajectory. 
        Each goal point is evaluated through the polynomial generated per reset. This specific
        implementation increases the number of polynomial evaluations by the epoch of training."""
        
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        self.approch_goal_metric = 0.5
        self.goal_generator = []

        # This value sets how many TOTAL goal points are to be evaluated during an episode. This does not include
        # the initial hover start point.
        self.goal_curriculum = 5.0

        #Tracks the required time between shifts in goal.
        self.goal_dt = []

        #Tracks the current time before a goal is changed. Init to all zeros.
        self.goal_time = [0] * self.num_agents

        
    def update_formation_size(self, new_formation_size):
        pass

    def step(self):
        self.update_formation_and_relate_param()

        tick = self.envs[0].tick
        
        time = self.envs[0].sim_steps*tick*(self.envs[0].dt) #  Current time in seconds.
        
        for i in range(self.num_agents):
            self.goal_time[i] += self.envs[0].sim_steps*self.envs[0].dt

            #change goals if we are within 1 time step
            if (abs(self.goal_time[i] - self.goal_dt[i]) < (self.envs[0].sim_steps*self.envs[0].dt)):

                next_goal = self.goal_generator[i].piecewise_eval(time)
        
                self.end_point[i] = next_goal.as_nparray()

                self.goals = copy.deepcopy(self.end_point)

                self.goal_time[i] = 0
            
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]

        if tick <= self.duration_step:
            return
       

        self.duration_step += int(self.envs[0].ep_time * self.envs[0].control_freq)
        
        return

    def reset(self, obst_map, cell_centers): 
  
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        
        if obst_map is None:
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
            traj_duration = np.random.uniform(low=2, high=self.envs[0].ep_time)

            # Generate trajectory with random time from (2, ep_time)
            self.goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=np.append(final_goal, np.random.uniform(low=0, high=3.14)), 
                                                   duration=traj_duration, current_time=0)
            
            self.goal_dt.append(traj_duration / self.goal_curriculum)

            #Find the initial goal
            self.end_point.append(self.goal_generator[i].piecewise_eval(0).as_nparray())
        
        self.duration_step = int(np.random.uniform(low=2.0, high=4.0) * self.envs[0].control_freq)
        self.update_formation_and_relate_param()

        self.formation_center = np.array((0., 0., 2.))
        self.spawn_points = copy.deepcopy(self.start_point)
        
        self.goals = copy.deepcopy(self.end_point)
        
        
        