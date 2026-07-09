from gym_art.quadrotor_multi.scenarios.obstacles.o_random import Scenario_o_random


def create_scenario(quads_mode, envs, num_agents, room_dims):
    if quads_mode != "o_random":
        raise ValueError(
            f"Unsupported scenario {quads_mode!r}. This repository only keeps the lidar o_random task."
        )
    return Scenario_o_random(quads_mode, envs, num_agents, room_dims)
