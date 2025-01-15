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
        self.approch_goal_metric = 1.0
        self.formation_list = ["circle", "grid"]

        # Position generation
        self.goal_generator = [QuadTrajGen(poly_degree=7) for _ in range(num_agents)]
        self.start_point = [np.zeros(3) for _ in range(num_agents)]
        self.end_point = [np.zeros(3) for _ in range(num_agents)]
        self.global_final_goals = [np.zeros(3) for _ in range(num_agents)]

        # The velocity of the trajectory is sampled from a normal distribution
        self.vel_mean = 0.5
        self.vel_std = 0.1

        self.min_speed = 0.3
        self.max_speed = 0.7
        self.max_duration_time = self.envs[0].ep_time - 2.0

        # Aux
        self.goal_scenario_flag = 0
        self.in_obst_area = 0
        self.obst_map = None
        self.cell_centers = None
        self.cur_duration = np.zeros(num_agents)

    def step(self):
        sim_steps = self.envs[0].sim_steps
        tick = self.envs[0].tick
        dt = self.envs[0].dt
        time = sim_steps * tick * dt

        if self.in_obst_area:
            if self.goal_scenario_flag:
                # Same goal
                if time >= max(self.cur_duration):
                    self.start_point = np.array(self.global_final_goals)
                    _, self.global_final_goals = self.generate_start_goal_pos_v2(
                        num_agents=self.num_agents, obst_map=self.obst_map, cell_centers=self.cell_centers,
                        goal_scenario_flag=self.goal_scenario_flag
                    )
                    self.update_traj_planner(current_time=time)
            else:
                for i in range(self.num_agents):
                    if time >= self.cur_duration[i]:
                        self.start_point[i] = np.array(self.global_final_goals[i])
                        self.global_final_goals[i] = self.sub_generate_pos_v2(
                            num_agents=1, obst_map=self.obst_map, cell_centers=self.cell_centers
                        )
                        self.update_traj_planner(current_time=time, update_agent_id=i)

        for i in range(self.num_agents):
            next_goal = self.goal_generator[i].piecewise_eval(time)
            self.end_point[i] = next_goal.as_nparray()

        self.goals = copy.deepcopy(self.end_point)
        for i, env in enumerate(self.envs):
            env.goal = self.end_point[i]
        
        return

    def update_traj_planner(self, current_time, update_agent_id=None):
        if self.goal_scenario_flag:
            # Same goal
            dist = np.linalg.norm(self.start_point[0] - self.global_final_goals[0])
            traj_speed = np.random.normal(self.vel_mean, self.vel_std)
            traj_speed = np.clip(traj_speed, a_min=self.min_speed, a_max=self.max_speed)
            traj_duration = dist / traj_speed
            traj_duration = min(traj_duration, self.max_duration_time)
            for i in range(self.num_agents):
                self.cur_duration[i] += traj_duration
                goal_yaw = 0
                initial_state = traj_eval()
                initial_state.set_initial_pos(self.start_point[i])
                self.goal_generator[i].plan_go_to_from(
                    initial_state=initial_state, desired_state=np.append(self.global_final_goals[i], goal_yaw),
                    duration=traj_duration, current_time=current_time
                )
        else:
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[update_agent_id])

            dist = np.linalg.norm(self.start_point[update_agent_id] - self.global_final_goals[update_agent_id])
            traj_speed = np.random.normal(self.vel_mean, self.vel_std)
            traj_speed = np.clip(traj_speed, a_min=self.min_speed, a_max=self.max_speed)
            traj_duration = dist / traj_speed
            traj_duration = min(traj_duration, self.max_duration_time)
            self.cur_duration[update_agent_id] += traj_duration
            goal_yaw = 0
            # Generate trajectory with random time from (0, ep_time)
            self.goal_generator[update_agent_id].plan_go_to_from(
                initial_state=initial_state, desired_state=np.append(self.global_final_goals[update_agent_id], goal_yaw),
                duration=traj_duration, current_time=current_time
            )

    def reset(self, params):
        transpose_obst_area_flag = params['transpose_obst_area_flag']
        self.obst_map = params['obst_map']
        self.cell_centers = params['cell_centers']
        # 0: Use different goal; 1: Use same goal
        self.goal_scenario_flag = np.random.choice([0, 1])
        # 0: From -x to x; 1: From x to -x
        pos_area_flag = np.random.choice([0, 1])
        self.in_obst_area = np.random.choice([0, 1])

        if self.goal_scenario_flag:
            self.approch_goal_metric = 1.0
        else:
            self.approch_goal_metric = 0.5

        # Find the goal point for all drones
        if self.goal_scenario_flag:
            # Same goal scenario
            formation_id = np.random.choice([0, len(self.formation_list) - 1])
            formation = self.formation_list[formation_id]
        else:
            formation = None

        if self.in_obst_area:
            self.start_point, self.global_final_goals = self.generate_start_goal_pos_v2(
                num_agents=self.num_agents, obst_map=self.obst_map, cell_centers=self.cell_centers,
                goal_scenario_flag=self.goal_scenario_flag
            )
        else:
            self.start_point, self.global_final_goals = self.generate_start_goal_pos(
                pos_area_flag=pos_area_flag, goal_scenario_flag=self.goal_scenario_flag, formation=formation,
                num_agents=self.num_agents, transpose_obst_area_flag=transpose_obst_area_flag
            )

        for i in range(self.num_agents):
            initial_state = traj_eval()
            initial_state.set_initial_pos(self.start_point[i])

            if self.goal_scenario_flag and self.in_obst_area:
                if i == 0:
                    # same goal and in obstacle area
                    dist_arr = np.linalg.norm(self.start_point - self.global_final_goals, axis=1)
                    dist = max(dist_arr)
                    traj_speed = np.random.normal(self.vel_mean, self.vel_std)
                    traj_speed = np.clip(traj_speed, a_min=self.min_speed, a_max=self.max_speed)
                    traj_duration = dist / traj_speed
                    # Clip in case the traj takes too long to finish.
                    traj_duration = min(traj_duration, self.max_duration_time)
                    self.cur_duration[i] = traj_duration
                else:
                    self.cur_duration[i] = self.cur_duration[0]
            else:
                # same goal and in obstacle area
                dist = np.linalg.norm(self.start_point[i] - self.global_final_goals[i])
                traj_speed = np.random.normal(self.vel_mean, self.vel_std)
                traj_speed = np.clip(traj_speed, a_min=self.min_speed, a_max=self.max_speed)
                traj_duration = dist / traj_speed
                # Clip in case the traj takes too long to finish.
                traj_duration = min(traj_duration, self.max_duration_time)
                self.cur_duration[i] = traj_duration

            goal_yaw = 0
            # Generate trajectory with random time from (0, ep_time)
            self.goal_generator[i].plan_go_to_from(
                initial_state=initial_state, desired_state=np.append(self.global_final_goals[i], goal_yaw),
                duration=float(self.cur_duration[i]), current_time=0
            )

            #Find the initial goal
            self.end_point[i] = self.goal_generator[i].piecewise_eval(0).as_nparray()

        self.spawn_points = copy.deepcopy(self.start_point)
        self.goals = copy.deepcopy(self.end_point)
        for i, env in enumerate(self.envs):
            env.dynamic_goal = True
