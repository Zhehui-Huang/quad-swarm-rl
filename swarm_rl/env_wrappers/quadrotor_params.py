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
                   choices=['xyz_vxyz_R_omega', 'xyz_vxyz_R_omega_floor', 'xyz_vxyz_R_omega_wall'],
                   help='obs space for quadrotor self')
    p.add_argument('--quads_episode_duration', default=15.0, type=float,
                   help='Override default value for episode duration')
    p.add_argument('--quads_encoder_type', default="corl", type=str, help='The type of the neighborhood encoder')

    # Neighbor
    # Neighbor Features
    p.add_argument('--quads_neighbor_visible_num', default=-1, type=int, help='Number of neighbors to consider. -1=all '
                                                                          '0=blind agents, '
                                                                          '0<n<num_agents-1 = nonzero number of agents')
    p.add_argument('--quads_neighbor_obs_type', default='none', type=str,
                   choices=['none', 'pos_vel'], help='Choose what kind of obs to send to encoder.')

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
    p.add_argument('--quads_collision_falloff_radius', default=-1.0, type=float,
                   help='The falloff radius for the smooth penalty. -1.0: no smooth penalty')
    p.add_argument('--quads_collision_smooth_max_penalty', default=10.0, type=float,
                   help='The upper bound of the collision function given distance among drones')
    p.add_argument('--quads_orient_reward', default=1.0, type=float, help='Reward weight for orientation stability')
    p.add_argument('--quads_spin_reward', default=0.1, type=float, help='Reward weight for angular velocity penalty')
    p.add_argument('--quads_vel_reward', default=0.0, type=float, help='Reward weight for high-speed penalty')
    p.add_argument('--quads_vel_penalty_limit', default=3.0, type=float,
                   help='Only penalize linear speed above this limit')
    p.add_argument('--quads_velocity_yaw_max_speed', default=3.0, type=float,
                   help='Maximum linear speed command for velocity_yaw control')
    p.add_argument('--quads_progress_reward', default=2.0, type=float,
                   help='Reward weight for moving toward the current goal')
    p.add_argument('--quads_success_reward', default=10.0, type=float,
                   help='One-step reward for reaching the current goal')
    p.add_argument('--quads_first_success_reward', default=10.0, type=float,
                   help='One-step reward for reaching the first goal in an episode')
    p.add_argument('--quads_z_reward', default=0.0, type=float, help='Reward weight for low-altitude penalty')
    p.add_argument('--quads_stable_z_reward', default=0.0, type=float, help='Reward weight for stable altitude bonus')
    p.add_argument('--quads_stable_spin_reward', default=0.0, type=float, help='Reward weight for stable spin bonus')
    p.add_argument('--quads_room_floor_reward', default=0.0, type=float, help='Penalty for hitting the floor boundary')
    p.add_argument('--quads_room_wall_reward', default=0.0, type=float, help='Penalty for hitting a side wall boundary')
    p.add_argument('--quads_room_ceiling_reward', default=0.0, type=float, help='Penalty for hitting the ceiling boundary')
    p.add_argument('--quads_control_type', default='velocity_yaw', type=str,
                   choices=[
                       'velocity_yaw', 'velocity_yaw_avoid', 'velocity_yaw_body_avoid',
                       'raw_motor', 'position',
                   ],
                   help='Low-level control interface used by the environment')
    p.add_argument('--quads_avoid_radius', default=0.8, type=float,
                   help='Lidar distance that activates CBF-QP velocity constraints')
    p.add_argument('--quads_cbf_safe_distance', default=0.5, type=float,
                   help='Minimum obstacle surface distance enforced by the CBF-QP safety filter')
    p.add_argument('--quads_cbf_alpha', default=1.5, type=float,
                   help='CBF class-K gain controlling how quickly safe clearance may decrease')
    p.add_argument('--quads_avoid_lidar_filter_alpha', default=0.35, type=float,
                   help='EMA update weight for lidar distances used by the CBF filter')
    p.add_argument('--quads_avoid_activation_hysteresis', default=0.08, type=float,
                   help='Extra clearance required before an active CBF constraint is disabled')
    p.add_argument('--quads_avoid_kp', default=1.4, type=float,
                   help='Deprecated PID setting retained for old checkpoint compatibility')
    p.add_argument('--quads_avoid_ki', default=0.15, type=float,
                   help='Deprecated PID setting retained for old checkpoint compatibility')
    p.add_argument('--quads_avoid_kd', default=0.25, type=float,
                   help='Deprecated PID setting retained for old checkpoint compatibility')
    p.add_argument('--quads_avoid_max_bias', default=1.2, type=float,
                   help='Deprecated PID setting retained for old checkpoint compatibility')
    p.add_argument('--quads_avoid_floor_guard_z', default=1.2, type=float,
                   help='Minimum altitude guarded by an avoidance controller')
    p.add_argument('--quads_avoid_floor_guard_kp', default=1.5, type=float,
                   help='Proportional gain for the avoidance low-altitude guard')
    p.add_argument('--quads_avoid_floor_guard_max_vz', default=0.8, type=float,
                   help='Maximum upward velocity command from the avoidance low-altitude guard')

    # Obstacle
    # # Obstacle Features
    p.add_argument('--quads_use_obstacles', default=False, type=str2bool, help='Use obstacles or not')
    p.add_argument('--quads_obstacle_obs_type', default='none', type=str,
                   choices=['none', 'octomap', 'lidar'], help='Choose what kind of obs to send to encoder.')
    p.add_argument('--quads_obst_density', default=0.2, type=float, help='Obstacle density in the map')
    p.add_argument('--quads_obst_size', default=1.0, type=float, help='Obstacle pillar diameter')
    p.add_argument('--quads_obst_spawn_area', nargs='+', default=[6.0, 6.0], type=float,
                   help='The spawning area of obstacles')
    p.add_argument('--quads_obst_min_clearance', default=0.0, type=float,
                   help='Minimum free distance between obstacle pillar surfaces')
    p.add_argument('--quads_obstacle_scan_resolution', default=0.1, type=float,
                   help='Resolution used for obstacle SDF samples around the quad. Lidar uses ray distances.')
    p.add_argument('--quads_lidar_sector_angle', default=0.0, type=float,
                   help='Angular width in degrees represented by each lidar direction.')
    p.add_argument('--quads_lidar_sector_samples', default=1, type=int,
                   help='Number of rays sampled inside each lidar sector.')
    p.add_argument('--quads_domain_random', default=False, type=str2bool, help='Use domain randomization or not')
    p.add_argument('--quads_obst_density_random', default=False, type=str2bool, help='Enable obstacle density randomization or not')
    p.add_argument('--quads_obst_density_min', default=0.05, type=float,
                   help='The minimum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_density_max', default=0.2, type=float,
                   help='The maximum of obstacle density when enabling domain randomization')
    p.add_argument('--quads_obst_size_random', default=False, type=str2bool, help='Enable obstacle size randomization or not')
    p.add_argument('--quads_obst_size_min', default=0.3, type=float,
                   help='The minimum obstacle size when enabling domain randomization')
    p.add_argument('--quads_obst_size_max', default=0.5, type=float,
                   help='The maximum obstacle size when enabling domain randomization')

    # # Obstacle Encoder
    p.add_argument('--quads_obst_hidden_size', default=256, type=int, help='The hidden size for the obstacle encoder')
    p.add_argument('--quads_obst_encoder_type', default='mlp', type=str, help='The type of the obstacle encoder')

    # # Obstacle Collision Reward
    p.add_argument('--quads_obst_collision_reward', default=0.0, type=float,
                   help='Override default value for quadcol_bin_obst reward, which means collisions between quadrotor '
                        'and obstacles')
    p.add_argument('--quads_floor_stall_reward', default=0.0, type=float,
                   help='Penalty applied when the quad stays on the floor')
    p.add_argument('--visualize_projection_map', default=False, type=str2bool,
                   help='Show a 2D projection map during evaluation')
    p.add_argument('--visualize_obstacle_point_cloud', default=False, type=str2bool,
                   help='Show nearby obstacle and wall point samples on the 2D projection map')

    # Aerodynamics
    # # Downwash
    p.add_argument('--quads_use_downwash', default=False, type=str2bool, help='Apply downwash or not')

    # Numba Speed Up
    p.add_argument('--quads_use_numba', default=False, type=str2bool, help='Whether to use numba for jit or not')

    # Scenarios
    p.add_argument('--quads_mode', default='o_random', type=str, choices=['o_random'],
                   help='Random waypoint navigation in the lidar obstacle field')

    # Room
    p.add_argument('--quads_room_dims', nargs='+', default=[10., 10., 10.], type=float,
                   help='Length, width, and height dimensions respectively of the quadrotor env')

    # Replay Buffer
    p.add_argument('--replay_buffer_sample_prob', default=0.0, type=float,
                   help='Probability at which we sample from it rather than resetting the env. Set to 0.0 (default) '
                        'to disable the replay. Set to value in (0.0, 1.0] to use replay buffer')

    # Annealing
    p.add_argument('--anneal_collision_steps', default=0.0, type=float, help='Anneal collision penalties over this '
                                                                             'many steps. Default (0.0) is no '
                                                                             'annealing')
    p.add_argument('--anneal_collision_initial_ratio', default=0.0, type=float,
                   help='Initial fraction of collision and velocity penalty weights during annealing')

    # Rendering
    p.add_argument('--quads_view_mode', nargs='+', default=['topdown', 'chase', 'global'],
                   type=str, choices=['topdown', 'chase', 'side', 'global', 'corner0', 'corner1', 'corner2', 'corner3', 'topdownfollow'],
                   help='Choose which kind of view/camera to use')
    p.add_argument('--quads_render', default=False, type=bool, help='Use render or not')
    p.add_argument('--visualize_v_value', action='store_true',
                   help='Legacy checkpoint option; V-value map visualization is not included')

    # Evaluation
    p.add_argument('--eval_num_episodes', default=50, type=int,
                   help='Number of deterministic episodes for obstacle traversal evaluation')
    p.add_argument('--eval_output_dir', default='eval_results', type=str,
                   help='Directory where obstacle traversal evaluation writes JSON and CSV summaries')
    p.add_argument('--eval_success_time_limit_s', default=45.0, type=float,
                   help='Maximum flight time for a successful obstacle traversal episode')
    p.add_argument('--eval_success_max_path_ratio', default=3.0, type=float,
                   help='Maximum path length divided by straight-line start-goal distance for success')
    p.add_argument('--eval_success_max_path_length_m', default=18.0, type=float,
                   help='Absolute path length cap for success in the 10 m room obstacle task')

    # Sim2Real
    p.add_argument('--quads_sim2real', default=False, type=str2bool, help='Whether to use sim2real or not')
