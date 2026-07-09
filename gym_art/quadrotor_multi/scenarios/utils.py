import numpy as np
from numpy import cos, sin
from numba import njit

QUADS_FORMATION_LIST = ['circle_horizontal']

QUADS_PARAMS_DICT = {
    'o_random': [['circle_horizontal'], [0.0, 0.0]],
}


def update_formation_and_max_agent_per_layer(mode):
    formation_index = np.random.randint(low=0, high=len(QUADS_PARAMS_DICT[mode][0]))
    formation = QUADS_FORMATION_LIST[formation_index]
    if formation.startswith("circle"):
        num_agents_per_layer = 8
    elif formation.startswith("grid"):
        num_agents_per_layer = 50
    else:
        # for 3D formations. Specific formations override this
        num_agents_per_layer = 8

    return formation, num_agents_per_layer


def update_layer_dist(low, high):
    layer_dist = np.random.uniform(low=low, high=high)
    return layer_dist


@njit
def spherical_coordinate(x, y):
    return [cos(x) * cos(y), sin(x) * cos(y), sin(y)]


@njit
def generate_points(n=3):
    if n < 3:
        # print("The number of goals can not smaller than 3, The system has cast it to 3")
        n = 3

    x = 0.1 + 1.2 * n

    pts = np.zeros((n, 3))
    start = (-1. + 1. / (n - 1.))
    increment = (2. - 2. / (n - 1.)) / (n - 1.)
    pi = np.pi
    for j in range(n):
        s = start + j * increment
        pts[j] = spherical_coordinate(
            x=s * x, y=pi / 2. * np.sign(s) * (1. - np.sqrt(1. - abs(s)))
        )
    return pts


@njit
def get_sphere_radius(num, dist):
    A = 1.75388487222762
    B = 0.860487305801679
    C = 10.3632729642351
    D = 0.0920858134405214
    ratio = (A - D) / (1 + (num / C) ** B) + D
    radius = dist / ratio
    return radius


@njit
def get_circle_radius(num, dist):
    theta = 2 * np.pi / num
    radius = (0.5 * dist) / np.sin(theta / 2)
    return radius


@njit
def get_grid_dim_number(num):
    sqrt_goal_num = np.sqrt(num)
    grid_number = int(np.floor(sqrt_goal_num))
    dim_1 = grid_number
    while dim_1 > 1:
        if num % dim_1 == 0:
            break
        else:
            dim_1 -= 1

    dim_2 = num // dim_1
    return dim_1, dim_2


def get_formation_range(mode, formation, num_agents, low, high, num_agents_per_layer):
    # Numba just makes it ~ 5 times slower
    if mode == 'swarm_vs_swarm':
        n = num_agents // 2
    else:
        n = num_agents

    if formation.startswith("circle"):
        formation_size_low = get_circle_radius(num_agents_per_layer, low)
        formation_size_high = get_circle_radius(num_agents_per_layer, high)
    elif formation.startswith("grid"):
        formation_size_low = low
        formation_size_high = high
    elif formation.startswith("sphere"):
        formation_size_low = get_sphere_radius(n, low)
        formation_size_high = get_sphere_radius(n, high)
    elif formation.startswith("cube"):
        formation_size_low = low
        formation_size_high = high
    else:
        raise NotImplementedError(f'{formation} is not supported!')

    return formation_size_low, formation_size_high


def get_goal_by_formation(formation, pos_0, pos_1, layer_pos=0.):
    # Numba just makes it ~ 4 times slower
    if formation.endswith("horizontal"):
        goal = np.array([pos_0, pos_1, layer_pos])
    elif formation.endswith("vertical_xz"):
        goal = np.array([pos_0, layer_pos, pos_1])
    elif formation.endswith("vertical_yz"):
        goal = np.array([layer_pos, pos_0, pos_1])
    else:
        raise NotImplementedError("Unknown formation")

    return goal
