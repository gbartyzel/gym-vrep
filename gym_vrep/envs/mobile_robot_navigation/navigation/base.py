import numpy as np


class Base(object):
    def __init__(self, target_position, wheel_diameter,
                 robot_width, dt):
        self._target_position = np.asarray(target_position)
        self._wheel_radius = wheel_diameter / 2.0
        self._body_width = robot_width
        self._dt = dt

        self._pose = np.zeros(3)
        self._polar_coordinates = np.zeros(2)

    def set_target_position(self, target_position):
        self._target_position = np.asarray(target_position)

    @property
    def pose(self):
        return self._pose

    @property
    def polar_coordinates(self):
        return np.round(self._polar_coordinates, 3)

    def compute_position(self, **kwargs):
        return NotImplementedError

    def reset(self, start_pose):
        return NotImplementedError

    @staticmethod
    def _angle_correction(angle):
        if angle >= 0:
            angle = np.fmod((angle + np.pi), (2 * np.pi)) - np.pi

        if angle < 0:
            angle = np.fmod((angle - np.pi), (2 * np.pi)) + np.pi

        return np.round(angle, 3)
