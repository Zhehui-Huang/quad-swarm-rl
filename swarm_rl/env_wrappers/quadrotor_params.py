from sample_factory.utils.utils import str2bool


def quadrotors_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mlp',
        encoder_subtype='mlp_quads',
        rnn_size=256,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
    )


# noinspection PyUnusedLocal
def add_quadrotors_env_args(env, parser):
    p = parser

    # Quadrotor features
    p.add_argument('--quads_num_agents', default=8, type=int, help='Override default value for the number of quadrotors')
    p.add_argument('--quads_obs_repr', default='xyz_vxyz_R_omega', type=str,
                   choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall', 'xyz_vxyz_yaw_omega'],
                   help='obs space for quadrotor self')
    p.add_argument('--quads_obs_rel_rot', default=False, type=str2bool, help='use relative rotation or not')
    p.add_argument('--quads_dynamic_goal', default=True, type=str2bool, help='use relative observations for all states. Only for dynamic goal scenarios.')
    p.add_argument('--quads_episode_duration', default=15.0, type=float,
                   help='Override default value for episode duration')
    p.add_argument('--quads_encoder_type', default='corl', choices=['corl', 'attention'], type=str,
                   help='The type of the neighborhood encoder')

    # Neighbor
    # Neighbor Features
    p.add_argument('--quads_neighbor_visible_num', default=-1, type=int,
                   help='Number of neighbors to consider. -1=all 0=blind agents, 0<n<num_agents-1 = nonzero number of '
                        'agents')
    p.add_argument('--quads_neighbor_obs_type', default='none', type=str, choices=['none', 'pos', 'pos_vel'],
                   help='Choose what kind of obs to send to encoder.')

    # # Neighbor Encoder
    p.add_argument('--quads_neighbor_hidden_size', default=256, type=int,
                   help='The hidden size for the neighbor encoder')
    p.add_argument('--quads_neighbor_encoder_type', default='attention', type=str,
                   choices=['attention', 'mean_embed', 'mlp', 'no_encoder'],
                   help='The type of the neighborhood encoder')

    # # Neighbor Collision Reward
    p.add_argument('--quads_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin reward, which means collisions between quadrotors')
    p.add_argument('--quads_collision_hitbox_radius', default=2.0, type=float,
                   help='When the distance between two drones are less than N arm_length, we would view them as '
                        'collide.')
    p.add_argument('--quads_collision_falloff_radius', default=4.0, type=float,
                   help='The falloff radius for the smooth penalty. -1.0: no smooth penalty')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float,
                   help='The upper bound of the collision function given distance among drones')

    # Obstacle
    # # Obstacle Features
    p.add_argument('--quads_use_obstacles', default=False, type=str2bool, help='Use obstacles or not')
    p.add_argument('--quads_obstacle_obs_type', default='none', type=str,
                   choices=['none', 'octomap', 'ToFs'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_obstacle_tof_resolution', default=4, type=int,
                   choices=[4, 8], help='Choose the resolution of ToFs, 4x4 or 8x8.')
    p.add_argument('--quads_obst_noise', default=0.0, type=float,
                   help='Standard deviation of Gaussian noise on obstacle observations')
    p.add_argument('--quads_obst_density', default=0.2, type=float, help='Obstacle density in the map')
    p.add_argument('--quads_obst_size', default=1.0, type=float, help='The diameter / length of obstacles')
    p.add_argument('--quads_obst_grid_size', default=1.0, type=float, help='The grid size that obstacles are placed on')
    p.add_argument('--quads_obst_spawn_area', nargs='+', default=[8.0, 8.0], type=float,
                   help='The spawning area of obstacles')
    p.add_argument('--quads_obst_spawn_center', default=True, type=str2bool,
                   help='Spawn obstacles at the center of grids or not')
    p.add_argument('--quads_obst_grid_size_random', default=False, type=str2bool,
                   help='Spawn obstacle grid size randomly or not')
    p.add_argument('--quads_obst_grid_size_range', nargs='+', default=[0.5, 0.8], type=float,
                   help='Spawn range of grid size')

    p.add_argument('--quads_domain_random', default=False, type=str2bool, help='Use domain randomization or not')
    p.add_argument('--quads_obst_density_random', default=False, type=str2bool, help='Enable obstacle density randomization or not')
    p.add_argument('--quads_obst_density_min', default=0.05, type=float,
                   help='The minimum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_density_max', default=0.2, type=float,
                   help='The maximum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_size_random', default=False, type=str2bool, help='Enable obstacle size randomization or not')
    p.add_argument('--quads_obst_size_min', default=0.3, type=float,
                   help='The minimum obstacle size when enabling domain randomization')
    p.add_argument('--quads_obst_size_max', default=0.6, type=float,
                   help='The maximum obstacle size when enabling domain randomization')

    # # Obstacle Encoder
    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the obstacle encoder')
    p.add_argument('--quads_obst_encoder_type', default='mlp', type=str, help='The type of the obstacle encoder')

    # # Obstacle Collision Reward
    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor '
                        'and obstacles')
    p.add_argument('--quads_obst_collision_smooth_max_penalty', default=0.0, type=float,
                   help='The upper bound of the collision function given distance among drones')
    p.add_argument('--quads_obst_collision_smooth_penalty_range', default=0.0, type=float,
                   help='The upper bound of the distance between obstacles and drones for smooth penalty to be applied')


    # Aerodynamics
    # # Downwash
    p.add_argument('--quads_use_downwash', default=False, type=str2bool, help='Apply downwash or not')

    # Numba Speed Up
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')

    # Scenarios
    p.add_argument('--quads_mode', default='static_same_goal', type=str,
                   choices=['static_same_goal', 'static_diff_goal', 'dynamic_same_goal', 'dynamic_diff_goal',
                            'ep_lissajous3D', 'ep_rand_bezier', 'swarm_vs_swarm', 'swap_goals', 'dynamic_formations',
                            'mix', 'o_uniform_same_goal_spawn', 'o_random',
                            'o_dynamic_diff_goal', 'o_dynamic_same_goal', 'o_diagonal', 'o_static_same_goal',
                            'o_static_diff_goal', 'o_swap_goals', 'o_ep_rand_bezier', 'o_random_dynamic_goal',
                            'o_random_dynamic_goal_curriculum', 'sim2real_A', 'sim2real_B'],
                   help='Choose which scenario to run. ep = evader pursuit')

    # Room
    p.add_argument('--quads_room_dims', nargs='+', default=[10., 10., 5.], type=float,
                   help='Length, width, and height dimensions respectively of the quadrotor env')

    # Replay Buffer
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float,
                   help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) '
                        'to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    # Annealing
    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this '
                                                                             'many steps. Default (0.0) is no '
                                                                             'annealing')

    # Rendering
    p.add_argument('--quads_view_mode', nargs='+', default=['topdown', 'chase', 'global'],
                   type=str, choices=['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow'],
                   help='Choose which kind of view/camera to use')
    p.add_argument('--quads_render', default=False, type=bool, help='Use render or not')
    p.add_argument('--quads_render_mode', default='human', type=str, choices=['human', 'rgb_array'], help='Rendering mode, V-value requires rgb_array')
    p.add_argument('--visualize_v_value', action='store_true', help="Visualize v value map")

    # Sim2Real
    p.add_argument('--quads_sim2real', default=False, type=str2bool, help='Whether to use sim2real or not')
