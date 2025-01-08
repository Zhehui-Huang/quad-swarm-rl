from gym_art.quadrotor_multi.quad_utils import normalize, cross, npa

class ChaseCamera(object):
    # for visualization.
    # a rough attempt at a reasonable third-person camera
    # that looks "over the quadrotor's shoulder" from behind
    def __init__(self, view_dist=4):
        self.view_dist = view_dist

    def reset(self, goal, pos, vel):
        self.goal = goal
        self.pos_smooth = pos
        self.vel_smooth = vel
        self.right_smooth, _ = normalize(cross(vel, npa(0, 0, 1)))

    def step(self, pos, vel):
        # lowpass filter
        ap = 0.6
        av = 0.8
        ar = 0.9
        self.pos_smooth = ap * self.pos_smooth + (1 - ap) * pos
        self.vel_smooth = av * self.vel_smooth + (1 - av) * vel
        self.pos = pos

        veln, n = normalize(self.vel_smooth)
        self.opp = -veln
        up = npa(0, 0, 1)
        ideal_vel, _ = normalize(self.goal - self.pos_smooth)
        # if True or np.abs(veln[2]) > 0.95 or n < 0.01 or np.dot(veln, ideal_vel) < 0.7:
        # look towards goal even though we are not heading there
        right, _ = normalize(cross(ideal_vel, up))
        # else:
        # right, _ = normalize(cross(veln, up))
        self.right_smooth = ar * self.right_smooth + (1 - ar) * right

    # return eye, center, up suitable for gluLookAt
    def look_at(self):
        up = npa(0, 0, 1)
        back, _ = normalize(cross(self.right_smooth, up))
        to_eye, _ = normalize(0.9 * self.opp + 0.2 * self.right_smooth)
        eye = self.pos_smooth + self.view_dist * (self.opp + 0.3 * up)
        center = self.pos_smooth
        return eye, center, up