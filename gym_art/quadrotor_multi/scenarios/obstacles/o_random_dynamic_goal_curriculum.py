
import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base
from gym_art.quadrotor_multi.quadrotor_traj_gen import QuadTrajGen
from gym_art.quadrotor_multi.quadrotor_planner import traj_eval

from sample_factory.envs.env_utils import TrainingInfoInterface

class Scenario_o_random_dynamic_goal_curriculum(Scenario_o_base, TrainingInfoInterface):
    """ This scenario implements a 13 dim goal that tracks a smooth polynomial trajectory. 
        Each goal point is evaluated through the polynomial generated per reset. The velocity increases
        per amount of training steps. """
        
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        TrainingInfoInterface.__init__(self)
        
        self.approch_goal_metric = 0.5

        self.goal_generator = [QuadTrajGen(poly_degree=7) for i in range(num_agents)]
        self.start_point = [np.zeros(3) for i in range(num_agents)]
        self.end_point = [np.zeros(3) for i in range(num_agents)]
        
        self.vel_mean = 0.25
        self.vel_std = 0.1

    def step(self):

        tick = self.envs[0].tick
        
        time = self.envs[0].sim_steps*tick*(self.envs[0].dt) #  Current time in seconds.
        
        for i in range(self.num_agents):

            next_goal = self.goal_generator[i].piecewise_eval(time)
            
            self.end_point[i] = next_goal.as_nparray()

        self.goals = copy.deepcopy(self.end_point)
            
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]
        
        return

    def reset(self, obst_map, cell_centers): 
        
        approx_total_training_steps = self.training_info.get('approx_total_training_steps', 0)

        
        self.obstacle_map = obst_map
        self.cell_centers = cell_centers
        
        if obst_map is None:
            raise NotImplementedError

        obst_map_locs = np.where(self.obstacle_map == 0)
        self.free_space = list(zip(*obst_map_locs))
        
        print("Curriculum Velocity Mean: {} at Global Step: {}".format(self.vel_mean, approx_total_training_steps))
            
        self.vel_mean = approx_total_training_steps / 800e6

        if (self.vel_mean > 0.8):
            self.vel_mean = 0.8
            
        if (self.vel_mean < 0.25):
            self.vel_mean = 0.25

        for i in range(self.num_agents):
            self.start_point[i] = self.generate_pos_obst_map()
            
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[i])
            
            final_goal = self.generate_pos_obst_map()
            
            # Fix the goal height at 0.65 m
            final_goal[2] = 0.65
            
            dist = np.linalg.norm(self.start_point[i] - final_goal)
            traj_speed = np.random.normal(self.vel_mean, self.vel_std)
            
            if (traj_speed < 0.25):
                traj_speed = 0.25

            traj_duration = dist / traj_speed
   
            goal_yaw = np.random.uniform(low=-3.14, high=3.14)

            self.goal_generator[i].plan_go_to_from(initial_state=initial_state, desired_state=np.append(final_goal, goal_yaw), 
                                                   duration=traj_duration, current_time=0)
            
            self.end_point[i] = self.goal_generator[i].piecewise_eval(0).as_nparray()

        self.spawn_points = copy.deepcopy(self.start_point)
        
        self.goals = copy.deepcopy(self.end_point)
              
        for i, env in enumerate(self.envs):
            env.dynamic_goal = True
        
        
        