import random
from collections import deque
from copy import deepcopy

import gymnasium as gym
import numpy as np


class ReplayBufferEvent:
    def __init__(self, env, obs):
        self.env = env
        self.obs = obs
        self.num_replayed = 0


class ReplayBuffer:
    def __init__(self, control_frequency, cp_step_size=0.5, buffer_size=20):
        self.control_frequency = control_frequency
        self.cp_step_size_sec = cp_step_size  # how often (seconds) a checkpoint is saved
        self.cp_step_size_freq = self.cp_step_size_sec * self.control_frequency
        self.buffer_idx = 0
        self.buffer = deque([], maxlen=buffer_size)

    def write_cp_to_buffer(self, env, obs):
        """
        A collision was found and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        """
        env.saved_in_replay_buffer = True

        # For example, replace the item with the lowest number of collisions in the last 10 replays
        evt = ReplayBufferEvent(env, obs)
        if len(self.buffer) < self.buffer.maxlen:
            self.buffer.append(evt)
        else:
            self.buffer[self.buffer_idx] = evt
        print(f"Added new collision event to buffer at {self.buffer_idx}")
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer.maxlen

    def sample_event(self):
        """
        Sample an event to replay
        """
        idx = random.randint(0, len(self.buffer) - 1)
        print(f'Replaying event at idx {idx}')
        self.buffer[idx].num_replayed += 1
        return self.buffer[idx]

    def cleanup(self):
        new_buffer = deque([], maxlen=self.buffer.maxlen)
        for event in self.buffer:
            if event.num_replayed < 10:
                new_buffer.append(event)

        self.buffer = new_buffer

    def avg_num_replayed(self):
        replayed_stats = [e.num_replayed for e in self.buffer]
        if not replayed_stats:
            return 0
        return np.mean(replayed_stats)

    def __len__(self):
        return len(self.buffer)


class ExperienceReplayWrapper(gym.Wrapper):
    def __init__(self, env, replay_buffer_sample_prob):
        super().__init__(env)
        self.replay_buffer = ReplayBuffer(env.envs[0].control_freq)
        self.replay_buffer_sample_prob = replay_buffer_sample_prob

        self.max_episode_checkpoints_to_keep = int(3.0 / self.replay_buffer.cp_step_size_sec)  # keep only checkpoints from the last 3 seconds
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        self.save_time_before_collision_sec = 1.5
        self.last_tick_added_to_buffer = -1e9

        # variables for tensorboard
        self.replayed_events = 0
        self.episode_counter = 0

    def save_checkpoint(self, obs):
        """
        Save a checkpoint every X steps so that we may load it later if a collision was found. This is NOT the same as the buffer
        Checkpoints are added to the buffer only if we find a collision and want to replay that event later on
        """
        self.episode_checkpoints.append((deepcopy(self.env), deepcopy(obs)))

    def reset(self):
        """For reset we just use the default implementation."""
        return self.env.reset()

    def step(self, action):
        obs, rewards, dones, infos = self.env.step(action)

        if any(dones):
            obs = self.new_episode()
            for i in range(len(infos)):
                if not infos[i]["episode_extra_stats"]:
                    infos[i]["episode_extra_stats"] = dict()

                tag = "replay"
                infos[i]["episode_extra_stats"].update({
                    f"{tag}/replay_rate": self.replayed_events / self.episode_counter,
                    f"{tag}/new_episode_rate": (self.episode_counter - self.replayed_events) / self.episode_counter,
                    f"{tag}/replay_buffer_size": len(self.replay_buffer),
                    f"{tag}/avg_replayed": self.replay_buffer.avg_num_replayed(),
                })

        else:
            use_active_replay_buffer = self.env.use_replay_buffer and self.env.activate_replay_buffer
            if use_active_replay_buffer and self.env.envs[0].tick % self.replay_buffer.cp_step_size_freq == 0:
                self.save_checkpoint(obs)

            collision_flag = self.env.last_step_unique_collisions.any()
            if self.env.use_obstacles:
                collision_flag = collision_flag or len(self.env.curr_quad_col) > 0

            start_tick_flag = self.env.envs[0].tick > self.env.collisions_grace_period_seconds * self.env.envs[0].control_freq
            gap_between_collisions = self.env.envs[0].tick - self.last_tick_added_to_buffer > 3 * self.env.envs[0].control_freq
            if collision_flag and use_active_replay_buffer and start_tick_flag and gap_between_collisions:
                steps_ago = int(self.save_time_before_collision_sec / self.replay_buffer.cp_step_size_sec)
                if steps_ago > len(self.episode_checkpoints):
                    print(f"Tried to read past the boundary of checkpoint_history. Steps ago: {steps_ago}, "
                          f"episode checkpoints: {len(self.episode_checkpoints)}, {self.env.envs[0].tick}")
                    raise IndexError
                else:
                    env, obs = self.episode_checkpoints[-steps_ago]
                    self.replay_buffer.write_cp_to_buffer(env, obs)
                    self.last_tick_added_to_buffer = self.env.envs[0].tick

        return obs, rewards, dones, infos

    def new_episode(self):
        """
        Normally this would go into reset(), but MultiQuadEnv is a multi-agent env that automatically resets.
        This means that reset() is never actually called externally and we need to take care of starting our new episode.
        """
        self.episode_counter += 1
        self.last_tick_added_to_buffer = -1e9
        self.episode_checkpoints = deque([], maxlen=self.max_episode_checkpoints_to_keep)

        use_replay_buffer = self.replay_buffer and self.env.activate_replay_buffer and len(self.replay_buffer) > 0
        sample_from_reply_buffer = np.random.uniform(0, 1) < self.replay_buffer_sample_prob
        if sample_from_reply_buffer and use_replay_buffer:
            self.replayed_events += 1
            event = self.replay_buffer.sample_event()
            env = event.env
            obs = event.obs
            replayed_env = deepcopy(env)
            replayed_env.scenes = self.env.scenes

            # we want to use these for tensorboard, so reset them to zero to get accurate stats
            replayed_env.collisions_per_episode = replayed_env.collisions_after_settle = 0
            replayed_env.obst_quad_collisions_per_episode = replayed_env.obst_quad_collisions_after_settle = 0
            self.env = replayed_env

            self.replay_buffer.cleanup()

            return obs
        else:
            obs = self.env.reset()
            self.env.saved_in_replay_buffer = False
            return obs
