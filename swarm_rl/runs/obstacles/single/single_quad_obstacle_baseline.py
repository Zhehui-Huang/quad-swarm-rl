QUAD_BASELINE_CLI_1 = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False '
    '--num_workers=72 --num_envs_per_worker=8 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--with_pbt=False --normalize_input=False --normalize_returns=False --reward_clip=10 '
    # Quad specific
    '--quads_use_numba=True --save_milestones_sec=10000 --anneal_collision_steps=0 '
    '--replay_buffer_sample_prob=0.75 '
    # Scenarios
    '--quads_mode=mix --quads_episode_duration=15.0 '
    # Self
    '--quads_obs_repr=xyz_vxyz_R_omega_floor '
    # Neighbor
    '--quads_neighbor_hidden_size=-1 --quads_neighbor_obs_type=none --quads_collision_hitbox_radius=2.0 '
    '--quads_collision_falloff_radius=4.0 --quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=4.0 '
    '--quads_neighbor_encoder_type=no_encoder --quads_neighbor_visible_num=0 '
    # Obstacles
    '--quads_use_obstacles=True --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.6 '
    '--quads_obst_collision_reward=5.0 --quads_obstacle_obs_type=octomap --quads_use_downwash=True'
)