import multiprocessing

DEFAULT_NUM_WORKERS = min(8, max(4, multiprocessing.cpu_count() // 2))

QUAD_BASELINE_CLI = (
    'python -m swarm_rl.train --env=quadrotor_multi --train_for_env_steps=10000000 --algo=APPO --use_rnn=False '
    f'--num_workers={DEFAULT_NUM_WORKERS} --num_envs_per_worker=4 --learning_rate=0.0001 --ppo_clip_value=5.0 --recurrence=1 '
    '--nonlinearity=tanh --actor_critic_share_weights=False --policy_initialization=xavier_uniform '
    '--adaptive_stddev=False --continuous_tanh_scale=1.2 --initial_stddev=0.35 '
    '--with_vtrace=False --max_policy_lag=100000000 --rnn_size=256 --with_pbt=False '
    '--gae_lambda=1.00 --max_grad_norm=5.0 --exploration_loss_coeff=0.0 --rollout=128 --batch_size=1024 '
    '--quads_use_numba=True --quads_num_agents=1 --quads_mode=o_random --quads_episode_duration=15.0 '
    '--quads_neighbor_encoder_type=no_encoder --quads_neighbor_hidden_size=0 --quads_neighbor_obs_type=none '
    '--quads_neighbor_visible_num=0 --replay_buffer_sample_prob=0.75 --anneal_collision_steps=8000000 '
    '--normalize_input=False --normalize_returns=False --reward_clip=10.0 --save_milestones_sec=3600 '
    '--quads_collision_reward=10.0 --quads_collision_smooth_max_penalty=8.0 --quads_obst_collision_reward=10.0 '
    '--quads_floor_stall_reward=10.0 --quads_room_floor_reward=10.0 --quads_room_wall_reward=10.0 '
    '--quads_room_ceiling_reward=10.0 --quads_orient_reward=2.0 --quads_spin_reward=0.5 '
    '--quads_z_reward=0.5 --quads_stable_z_reward=0.5 --quads_stable_spin_reward=0.5'
)
