import numpy as np
from dataclasses import dataclass

class QuadPlanner:
    """
    This class implements a piecewise polynomial trajectory representation for a differentially flat quadrotor representation.
    See: D. Mellinger and V. Kumar, "Minimum snap trajectory generation and control for quadrotors"
    """
    def __init__ (
        self,
        poly_degree: int = 7
    ):
        """ Creates the quad planner class.

        Arguments:
            poly_degree: polynomial degree for trajectory representation. 
        """
        self.poly_degree = poly_degree
        self.planned_trajectory = {
            # Piecewise Trajectory
            "t_begin": 0.0,
            "n_pieces": 0,
            "pieces": [poly4d(degree=self.poly_degree)]
            # "pieces": []
        }

@dataclass
class poly4d:
    """ Dataclass for a single basis function with a state dimension of 4: x-y-z-yaw """
    degree: int = 7
    poly: np.ndarray = np.empty([4, degree + 1])
    duration: float = 0

@dataclass
class traj_eval:
    """ Holds data for a goal point 
        pos: [x,y,z] (m)
        vel: [x,y,z] (m/s)
        acc: [x,y,z] (m/s^2)
        omega: [roll, pitch, yaw] (rad/s)
        yaw: Radians
        NOTE: All values default to zero.
    """
    pos: np.ndarray = np.zeros(3) 
    vel: np.ndarray = np.zeros(3)
    acc: np.ndarray = np.zeros(3)
    omega: np.ndarray = np.zeros(3)
    yaw: float = 0

    def set_initial_state(self, state: np.ndarray):
        """ Sets the corresponding states based on input state: [x,y,z,yaw] """
        self.pos[0] = state[0]
        self.pos[1] = state[1]
        self.pos[2] = state[2]

        self.yaw = state[3]

    


        