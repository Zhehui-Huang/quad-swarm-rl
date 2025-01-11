import numpy as np

class TopDownCamera(object):
    def __init__(self, view_dist=3.0):
        self.radius = view_dist
        self.theta = np.pi / 2
        self.phi = 0.0
        self.height = 11.0
        self.center = np.array([0., 0., self.height])

    def reset(self, view_dist=2.0, center=np.array([0., 0., 5.])):
        self.center = np.array([0., 0., self.height])

    def step(self, center=np.array([0., 0., 2.])):
        pass

    def look_at(self):
        up = np.array([0, 1, 0])
        eye = self.center  # pattern center
        center = self.center - np.array([0, 0, 2])
        center = (center/np.linalg.norm(center)) * self.radius
        return eye, center, up