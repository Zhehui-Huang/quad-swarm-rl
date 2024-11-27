import numpy as np
import copy

from gym_art.quadrotor_multi.scenarios.obstacles.o_base import Scenario_o_base
from gym_art.quadrotor_multi.quadrotor_traj_gen import QuadTrajGen
from gym_art.quadrotor_multi.quadrotor_planner import traj_eval


class Scenario_o_random_dynamic_goal(Scenario_o_base):
    """ This scenario implements a 13 dim goal that tracks a smooth polynomial trajectory. 
        Each goal point is evaluated through the polynomial generated per reset."""
        
    def __init__(self, quads_mode, envs, num_agents, room_dims):
        super().__init__(quads_mode, envs, num_agents, room_dims)
        # Preset
        self.approch_goal_metric = 0.2 * self.num_agents
        self.formation_list = ["circle", "grid"]

        # Position generation
        self.goal_generator = [QuadTrajGen(poly_degree=7) for _ in range(num_agents)]
        self.start_point = [np.zeros(3) for _ in range(num_agents)]
        self.end_point = [np.zeros(3) for _ in range(num_agents)]
        self.global_final_goals = [np.zeros(3) for _ in range(num_agents)]

        # The velocity of the trajectory is sampled from a normal distribution
        self.vel_mean = 0.35
        self.vel_std = 0.1

        # Aux
        self.goal_scenario_flag = 0
        self.in_obst_area = 0

    def step(self):
        sim_steps = self.envs[0].sim_steps
        tick = self.envs[0].tick
        dt = self.envs[0].dt
        time = sim_steps * tick * dt

        for i in range(self.num_agents):
            next_goal = self.goal_generator[i].piecewise_eval(time)
            self.end_point[i] = next_goal.as_nparray()
            
        self.goals = copy.deepcopy(self.end_point)

        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]
        
        return

    def reset(self, obst_map=None, cell_centers=None, sim2real_scenario=False):
        # 0: Use different goal; 1: Use same goal
        self.goal_scenario_flag = np.random.choice([0, 1])
        # self.goal_scenario_flag = 1
        # 0: From -x to x; 1: From x to -x
        pos_area_flag = np.random.choice([0, 1])

        # Find the goal point for all drones
        if self.goal_scenario_flag:
            # Same goal scenario
            formation_id = np.random.choice([0, len(self.formation_list) - 1])
            formation = self.formation_list[formation_id]
        else:
            formation = None

        self.in_obst_area = np.random.choice([0, 1])
        if self.in_obst_area:
            self.start_point, _ = self.generate_start_goal_pos_v2(
                pos_area_flag=pos_area_flag, goal_scenario_flag=self.goal_scenario_flag, formation=formation,
                num_agents=self.num_agents
            )
            self.obstacle_map = obst_map
            self.cell_centers = cell_centers
            if self.obstacle_map is None:
                raise NotImplementedError
            else:
                obst_map_locs = np.where(self.obstacle_map == 0)
                self.free_space = list(zip(*obst_map_locs))

            if self.goal_scenario_flag:
                # same goal
                self.global_final_goals = self.generate_goal_same(check_surroundings=True)
            else:
                self.global_final_goals = self.generate_goal_diff(num_agents=self.num_agents)
        else:
            self.start_point, self.global_final_goals = self.generate_start_goal_pos(
                pos_area_flag=pos_area_flag, goal_scenario_flag=self.goal_scenario_flag, formation=formation,
                num_agents=self.num_agents
            )

        for i in range(self.num_agents):
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[i])

            dist = np.linalg.norm(self.start_point[i] - self.global_final_goals[i])
            traj_speed = np.random.normal(self.vel_mean, self.vel_std)
            traj_speed = np.clip(traj_speed, a_min=0.2, a_max=0.4)
            traj_duration = dist / traj_speed
            # Clip in case the traj takes too long to finish.
            traj_duration = np.clip(traj_duration, a_min=0.5, a_max=self.envs[0].ep_time - 0.5)

            goal_yaw = 0
            # Generate trajectory with random time from (0, ep_time)
            self.goal_generator[i].plan_go_to_from(
                initial_state=initial_state, desired_state=np.append(self.global_final_goals[i], goal_yaw),
                duration=traj_duration, current_time=0
            )

            #Find the initial goal
            self.end_point[i] = self.goal_generator[i].piecewise_eval(0).as_nparray()

        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = copy.deepcopy(self.end_point)
        for i, env in enumerate(self.envs):
            env.dynamic_goal = True
