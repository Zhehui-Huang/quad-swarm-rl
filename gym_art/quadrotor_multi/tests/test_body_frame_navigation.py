import math
from types import SimpleNamespace
from unittest import TestCase

import gymnasium as gym
import numpy as np
import torch

from gym_art.quadrotor_multi.get_state import _policy_frame_position_velocity
from gym_art.quadrotor_multi.quadrotor_control import VelocityYawAvoidControl
from gym_art.quadrotor_multi.scenarios.obstacles.o_random import Scenario_o_random
from swarm_rl.bootstrap_lidar_v6 import build_finetune_checkpoint
from swarm_rl.env_wrappers.quad_utils import AnnealSchedule
from swarm_rl.env_wrappers.quad_utils import BoundedActionWrapper
from swarm_rl.env_wrappers.reward_shaping import QuadsRewardShapingWrapper


class TestBodyFrameObservation(TestCase):
    def test_rotates_goal_and_velocity_into_body_yaw_frame(self):
        env = SimpleNamespace(
            control_type="velocity_yaw_body_avoid",
            goal=np.zeros(3),
            dynamics=SimpleNamespace(
                rot=np.array([
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ])
            ),
        )

        relative_pos, velocity = _policy_frame_position_velocity(
            env,
            pos=np.array([1.0, 0.0, 2.0]),
            vel=np.array([0.0, 1.0, -0.5]),
        )

        np.testing.assert_allclose(relative_pos, [0.0, -1.0, 2.0], atol=1e-7)
        np.testing.assert_allclose(velocity, [1.0, 0.0, -0.5], atol=1e-7)

    def test_leaves_world_frame_control_observation_unchanged(self):
        env = SimpleNamespace(
            control_type="velocity_yaw_avoid",
            goal=np.array([0.5, -0.5, 1.0]),
            dynamics=SimpleNamespace(rot=np.eye(3)),
        )

        relative_pos, velocity = _policy_frame_position_velocity(
            env,
            pos=np.array([1.5, 0.5, 2.0]),
            vel=np.array([0.2, -0.3, 0.4]),
        )

        np.testing.assert_allclose(relative_pos, [1.0, 1.0, 1.0])
        np.testing.assert_allclose(velocity, [0.2, -0.3, 0.4])


class TestCbfQpSafetyFilter(TestCase):
    def setUp(self):
        self.controller = VelocityYawAvoidControl.__new__(VelocityYawAvoidControl)
        self.controller.max_speed = 1.2
        self.controller.avoid_radius = 1.2
        self.controller.cbf_safe_distance = 0.5
        self.controller.cbf_alpha = 1.5
        self.controller.lidar_filter_alpha = 1.0
        self.controller.activation_hysteresis = 0.1
        self.controller.reset()

    def test_blocks_velocity_toward_obstacle_at_safe_boundary(self):
        lidar = 100.0 * np.ones(9)
        lidar[0] = self.controller.cbf_safe_distance

        safe_velocity = self.controller._safe_body_velocity(np.array([1.2, 0.0]), lidar)

        np.testing.assert_allclose(safe_velocity, [0.0, 0.0], atol=1e-7)

    def test_keeps_unconstrained_lateral_velocity(self):
        lidar = 100.0 * np.ones(9)
        lidar[0] = self.controller.cbf_safe_distance

        safe_velocity = self.controller._safe_body_velocity(np.array([1.2, 0.4]), lidar)

        np.testing.assert_allclose(safe_velocity, [0.0, 0.4], atol=1e-7)

    def test_uses_ninth_sector_as_directional_constraint(self):
        lidar = 100.0 * np.ones(9)
        lidar[8] = self.controller.cbf_safe_distance
        direction = self.controller._lidar_directions()[8]

        safe_velocity = self.controller._safe_body_velocity(1.2 * direction, lidar)

        self.assertLessEqual(float(np.dot(direction, safe_velocity)), 1e-7)

    def test_hysteresis_keeps_constraint_active_near_activation_radius(self):
        lidar = 100.0 * np.ones(9)
        lidar[0] = self.controller.avoid_radius - 0.01
        self.controller._safe_body_velocity(np.array([1.2, 0.0]), lidar)

        lidar[0] = self.controller.avoid_radius + 0.05
        self.controller._safe_body_velocity(np.array([1.2, 0.0]), lidar)
        self.assertTrue(self.controller.active_lidar_constraints[0])

        lidar[0] = self.controller.avoid_radius + 0.11
        self.controller._safe_body_velocity(np.array([1.2, 0.0]), lidar)
        self.assertFalse(self.controller.active_lidar_constraints[0])

    def test_lidar_filter_rejects_single_step_distance_jump(self):
        self.controller.lidar_filter_alpha = 0.25
        lidar = 100.0 * np.ones(9)
        lidar[0] = 0.6
        self.controller._safe_body_velocity(np.array([1.2, 0.0]), lidar)

        lidar[0] = 1.4
        self.controller._safe_body_velocity(np.array([1.2, 0.0]), lidar)

        self.assertAlmostEqual(self.controller.filtered_lidar[0], 0.8)
        self.assertTrue(self.controller.active_lidar_constraints[0])


class TestBoundedActionWrapper(TestCase):
    def test_clips_action_before_forwarding_to_environment(self):
        class RecordingEnv(gym.Env):
            action_space = gym.spaces.Box(
                low=np.array([-1.0, -2.0], dtype=np.float32),
                high=np.array([1.0, 2.0], dtype=np.float32),
            )
            observation_space = gym.spaces.Box(-np.ones(1), np.ones(1), dtype=np.float32)

            def step(self, action):
                self.last_action = action
                return None, 0.0, False, {}

        env = RecordingEnv()
        BoundedActionWrapper(env).step(np.array([5.0, -7.0]))

        np.testing.assert_allclose(env.last_action, [1.0, -2.0])

    def test_reward_wrapper_accepts_gymnasium_reset_arguments(self):
        class LegacyEnv(gym.Env):
            num_agents = 1
            rew_coeff = {"pos": 0.1}

            def reset(self):
                return np.array([1.0])

        env = QuadsRewardShapingWrapper(
            LegacyEnv(),
            reward_shaping_scheme={"quad_rewards": {"pos": 0.2}},
        )

        obs = env.reset(seed=7, options={"unused": True})

        np.testing.assert_allclose(obs, [1.0])
        self.assertEqual(env.unwrapped.rew_coeff["pos"], 0.2)


class TestV6FinetuneBootstrap(TestCase):
    def test_resets_progress_optimizer_and_action_stddev(self):
        source = {
            "train_step": 123,
            "env_steps": 456,
            "best_performance": 42.0,
            "curr_lr": 1e-4,
            "model": {
                "encoder.weight": torch.ones(2),
                "action_parameterization.learned_stddev": torch.zeros(4),
            },
            "optimizer": {
                "state": {0: {"exp_avg": torch.ones(1)}},
                "param_groups": [{"lr": 1e-4, "params": [0]}],
            },
        }

        checkpoint = build_finetune_checkpoint(source, action_stddev=0.30, learning_rate=3e-5)

        self.assertEqual(checkpoint["train_step"], 0)
        self.assertEqual(checkpoint["env_steps"], 0)
        self.assertEqual(checkpoint["best_performance"], -1e9)
        self.assertEqual(checkpoint["optimizer"]["state"], {})
        self.assertEqual(checkpoint["optimizer"]["param_groups"][0]["lr"], 3e-5)
        torch.testing.assert_close(
            checkpoint["model"]["action_parameterization.learned_stddev"],
            torch.full((4,), math.log(0.30)),
        )
        torch.testing.assert_close(checkpoint["model"]["encoder.weight"], torch.ones(2))

    def test_anneal_schedule_starts_above_zero_and_reaches_target(self):
        schedule = AnnealSchedule("collision", final_value=8.0, anneal_env_steps=5_000_000, initial_value=1.0)

        self.assertEqual(schedule.value_at(0), 1.0)
        self.assertAlmostEqual(schedule.value_at(1_000_000), 2.4)
        self.assertAlmostEqual(schedule.value_at(5_000_000), 8.0)
        self.assertAlmostEqual(schedule.value_at(6_000_000), 8.0)


class TestRandomWaypointRefresh(TestCase):
    def test_step_reports_goal_change(self):
        env = SimpleNamespace(
            dynamics=SimpleNamespace(pos=np.zeros(3)),
            goal=np.zeros(3),
            rew_coeff={"success": 10.0},
        )
        scenario = Scenario_o_random("o_random", [env], 1, [12.0, 12.0, 10.0])
        scenario.goals = np.array([[0.1, 0.0, 0.0]])
        scenario.generate_new_goal_for_agent = lambda _: np.array([4.0, 0.0, 5.0])

        self.assertTrue(scenario.step())
        np.testing.assert_allclose(env.goal, [4.0, 0.0, 5.0])
        self.assertEqual(env.rew_coeff["success"], scenario.goal_success_later)

    def test_step_reports_no_change_away_from_goal(self):
        env = SimpleNamespace(
            dynamics=SimpleNamespace(pos=np.zeros(3)),
            goal=np.zeros(3),
            rew_coeff={"success": 10.0},
        )
        scenario = Scenario_o_random("o_random", [env], 1, [12.0, 12.0, 10.0])
        scenario.goals = np.array([[2.0, 0.0, 0.0]])

        self.assertFalse(scenario.step())
        np.testing.assert_allclose(env.goal, [2.0, 0.0, 0.0])
