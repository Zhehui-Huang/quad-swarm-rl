import numpy as np
from gym_art.quadrotor_multi.quad_utils import normalize, cross

class TopDownFollowCamera(object):
    def __init__(self, view_dist=4.0):
        self.view_dist = view_dist
        self.goal = None
        self.pos_smooth = None
        self.vel_smooth = None
        self.right_smooth = None

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, np.array([0, 0, 1])))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = np.array([0, 1, 0])
        eye = self.pos_smooth + np.array([0, 0, 5])
        center = self.pos_smooth
        return eye, center, up