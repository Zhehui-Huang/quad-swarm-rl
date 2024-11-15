import numpy as np

def get_circle_radius(num_agents, dist):
    radius = dist / (2 * np.sin(np.pi / num_agents))
    return radius

def get_goals_given_formation(formation, dist_range, formation_center, num_agents):
    pi = np.pi
    goals = []

    if formation == 'circle':
        dist_min, dist_max = dist_range
        dist = np.random.uniform(dist_min, dist_max)
        formation_size = get_circle_radius(num_agents=num_agents, dist=dist)

        for i in range(num_agents):
            degree = 2 * pi * i / num_agents
            pos_0 = formation_size * np.cos(degree)
            pos_1 = formation_size * np.sin(degree)

            pos_0 = round(pos_0, 3)
            pos_1 = round(pos_1, 3)

            goal = np.array([pos_0, pos_1, 0.0])
            goals.append(goal)

        goals = np.array(goals)
        goals += formation_center
    elif formation == 'grid':
        num_cols = int(np.ceil(np.sqrt(num_agents)))
        num_rows = int(np.ceil(num_agents / num_cols))

        dist_min, dist_max = dist_range
        dist = np.random.uniform(dist_min, dist_max)

        count = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if count >= num_agents:
                    break

                x = round(row * dist, 3)
                y = round(col * dist, 3)
                goal = np.array([x, y, 0.0])
                goals.append(goal)
                count += 1

        mean_pos = np.mean(goals, axis=0)
        goals = np.array(goals) - mean_pos + formation_center
    else:
        raise ValueError(f"Formation {formation} not supported.")

    return goals


if __name__ == '__main__':
    formation_center = np.array([0., 0., 0.])
    formation = 'grid'
    dist_range = [0.3, 0.6]
    num_agents = 8

    # d_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # for d in d_list:
    #     radius = get_circle_radius(num_agents=num_agents, dist=d)
    #     print(f"Distance: {d}, Radius: {radius}")

    goals = get_goals_given_formation(
        formation=formation, dist_range=dist_range, formation_center=formation_center, num_agents=num_agents
    )
    print(goals)
