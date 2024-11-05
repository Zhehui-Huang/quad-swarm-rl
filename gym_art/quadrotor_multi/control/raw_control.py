from gymnasium import spaces
from gym_art.quadrotor_multi.quad_utils import *

GRAV = 9.81


class RawControl(object):
    def __init__(self, dynamics, zero_action_middle=True):
        self.zero_action_middle = zero_action_middle
        self.action = None
        self.step_func = self.step
        self.low = np.zeros(4)
        self.high = np.ones(4)
        self.bias = 0.0
        self.scale = 1.0

    def action_space(self, dynamics):
        if not self.zero_action_middle:
            # Range of actions 0 .. 1
            self.low = np.zeros(4)
            self.bias = 0.0
            self.scale = 1.0
        else:
            # Range of actions -1 .. 1
            self.low = -np.ones(4)
            self.bias = 1.0
            self.scale = 0.5
        self.high = np.ones(4)
        return spaces.Box(self.low, self.high, dtype=np.float32)


    def step(self, dynamics, action, goal, dt, observation=None):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()

    def step_tf(self, dynamics, action, goal, dt, observation=None):
        action = np.clip(action, a_min=self.low, a_max=self.high)
        action = self.scale * (action + self.bias)
        dynamics.step(action, dt)
        self.action = action.copy()

