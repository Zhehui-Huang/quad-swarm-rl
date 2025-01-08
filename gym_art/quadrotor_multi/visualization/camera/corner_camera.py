import numpy as np

class CornerCamera(object):
    def __init__(self, view_dist=4.0, room_dims=np.array([10, 10, 10]), corner_index=0):
        self.radius = view_dist
        self.theta = np.pi / 2
        self.phi = 0.0
        self.center = np.array([0., 0., 2.])
        self.corner_index = corner_index
        self.room_dims = room_dims
        if corner_index == 0:
            self.center = np.array([-self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif corner_index == 1:
            self.center = np.array([self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif corner_index == 2:
            self.center = np.array([-self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])
        elif corner_index == 3:
            self.center = np.array([self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])

    def reset(self, view_dist=4.0, center=None):
        if center is not None:
            self.center = center
        elif self.corner_index == 0:
            self.center = np.array([-self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif self.corner_index == 1:
            self.center = np.array([self.room_dims[0] / 2, -self.room_dims[1] / 2, self.room_dims[2]])
        elif self.corner_index == 2:
            self.center = np.array([-self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])
        elif self.corner_index == 3:
            self.center = np.array([self.room_dims[0] / 2, self.room_dims[1] / 2, self.room_dims[2]])

    def step(self, center=np.array([0., 0., 2.])):
        pass

    def look_at(self):
        up = np.array([0, 0, 1])
        eye = self.center  # pattern center
        center = self.center - np.array([0, 0, 2])
        center = (center/np.linalg.norm(center)) * self.radius
        return eye, center, up

