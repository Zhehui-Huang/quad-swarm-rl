QUAD_BASELINE_CLI_8 = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=1000000000 --algo=APPO --use_rnn=False '
    '--num_workers=36 --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --with_vtrace=False --max_policy_lag=100000000 --rnn_size=16 '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--with_pbt=False --normalize_input=False --normalize_returns=False --reward_clip=10 '
    # Quad specific
    '--quads_use_numba=True --save_milestones_sec=10000 --anneal_collision_steps=0 '
    '--replay_buffer_sample_prob=0.0 '
    # Scenarios
    '--quads_mode=mix --quads_episode_duration=15.0 --quads_use_downwash=True '
    # Self
    '--quads_obs_repr=xyz_vxyz_R_omega --quads_sim2real=True --quads_encoder_type=attention --quads_num_agents=8 '
    # Neighbor
    '--quads_neighbor_hidden_size=16 --quads_neighbor_obs_type=pos --quads_collision_hitbox_radius=2.0 '
    '--quads_collision_falloff_radius=4.0 --quads_collision_reward=5.0 --quads_collision_smooth_max_penalty=4.0 '
    '--quads_neighbor_encoder_type=no_encoder --quads_neighbor_visible_num=2 '
    # Obstacles
    '--quads_use_obstacles=True --quads_obst_spawn_area 8 8 --quads_obst_density=0.2 --quads_obst_size=0.3 '
    '--quads_obst_collision_reward=5.0 --quads_obstacle_obs_type=ToFs'
)