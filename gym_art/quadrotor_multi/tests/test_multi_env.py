import time
from unittest import TestCase
import numpy as np

from gym_art.quadrotor_multi.quad_experience_replay import ExperienceReplayWrapper
from gym_art.quadrotor_multi.quadrotor_multi import QuadrotorEnvMulti
from swarm_rl.env_wrappers.reward_shaping import DEFAULT_QUAD_REWARD_SHAPING


def create_env(num_agents, use_numba=False, use_replay_buffer=False, episode_duration=7, local_obs=-1):
    quad = 'Crazyflie'
    dyn_randomize_every = dyn_randomization_ratio = None

    episode_duration = episode_duration  # seconds

    raw_control = raw_control_zero_middle = True

    sampler_1 = None
    if dyn_randomization_ratio is not None:
        sampler_1 = dict(type="RelativeSampler", noise_ratio=dyn_randomization_ratio, sampler="normal")

    sense_noise = 'default'

    dynamics_change = dict(noise=dict(thrust_noise_ratio=0.05), damp=dict(vel=0, omega_quadratic=0))

    env = QuadrotorEnvMulti(
        num_agents=num_agents, ep_time=episode_duration, rew_coeff=DEFAULT_QUAD_REWARD_SHAPING['quad_rewards'],
        obs_repr='xyz_vxyz_R_omega', obs_rel_rot=False, neighbor_visible_num=local_obs, neighbor_obs_type='pos_vel',
        collision_hitbox_radius=2.0, collision_falloff_radius=-1.0, use_obstacles=True, obst_density=0.2, obst_size=0.3,
        obst_spawn_area=[6.0, 6.0], obst_obs_type='ToFs', obst_noise=0.0, grid_size=1.0,
        use_downwash=False, use_numba=use_numba, quads_mode='o_random', room_dims=[10., 10., 5.],
        use_replay_buffer=use_replay_buffer, quads_view_mode=['topdown', 'chase', 'global'], quads_render=True,

        dynamics_params=quad, raw_control=raw_control, raw_control_zero_middle=raw_control_zero_middle,
        dynamics_randomize_every=dyn_randomize_every, dynamics_change=dynamics_change, dyn_sampler_1=sampler_1,
        sense_noise=sense_noise, init_random_state=True,
        render_mode='human',
    )
    return env


class TestMultiEnv(TestCase):
    def test_basic(self):
        num_agents = 2
        env = create_env(num_agents, use_numba=False)

        self.assertTrue(hasattr(env, 'num_agents'))
        self.assertEqual(env.num_agents, num_agents)

        obs = env.reset()
        self.assertIsNotNone(obs)

        for i in range(100):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])
            try:
                self.assertIsInstance(obs, list)
            except:
                self.assertIsInstance(obs, np.ndarray)

            self.assertIsInstance(rewards, list)
            self.assertIsInstance(dones, list)
            self.assertIsInstance(infos, list)

        env.close()

    def test_render(self):
        num_agents = 16
        env = create_env(num_agents, use_numba=False, local_obs=8)
        env.render_speed = 1.0

        env.reset()
        time.sleep(0.1)

        num_steps = 0
        render_n_frames = 100

        render_start = None
        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()

            if num_steps <= 1:
                render_start = time.time()

        render_took = time.time() - render_start
        print(f"Rendering of {render_n_frames} frames took {render_took:.3f} sec")

        env.close()

    def test_local_info(self):
        num_agents = 16
        env = create_env(num_agents, use_numba=False, local_obs=8)

        env.reset()

        for i in range(100):
            obs, rewards, dones, infos = env.step([env.action_space.sample() for i in range(num_agents)])

        env.close()


class TestReplayBuffer(TestCase):
    def test_replay(self):
        num_agents = 16
        replay_buffer_sample_prob = 1.0
        env = create_env(num_agents, use_numba=False, use_replay_buffer=replay_buffer_sample_prob > 0, episode_duration=5)
        env.render_speed = 1.0
        env = ExperienceReplayWrapper(env, replay_buffer_sample_prob=replay_buffer_sample_prob, default_obst_density=0.2, defulat_obst_size=0.3)

        env.reset()
        time.sleep(0.01)

        num_steps = 0
        render_n_frames = 150

        while num_steps < render_n_frames:
            obs, rewards, dones, infos = env.step([env.action_space.sample() for _ in range(num_agents)])
            num_steps += 1
            # print('Rewards: ', rewards, "\nCollisions: \n", env.collisions, "\nDistances: \n", env.dist)
            env.render()
            # this env self-resets

        env.close()
