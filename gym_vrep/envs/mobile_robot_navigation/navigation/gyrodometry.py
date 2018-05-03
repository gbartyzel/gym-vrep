import numpy as np

from gym_vrep.envs.mobile_robot_navigation.navigation.base import Base


class Gyrodometry(Base):
    def __init__(self, target_position, wheel_diameter,
                 robot_width, dt):
        super(Gyrodometry, self).__init__(target_position,
                                          wheel_diameter,
                                          robot_width, dt)

        self._sum_path = 0.0

        self._previous_phi = 0.0
        self._previous_angular_velocity = 0.0

    @property
    def sum_path(self):
        return self._sum_path

    def compute_position(self, **kwargs):
        delta_path = self.compute_delta_motion(kwargs['phi'])
        delta_beta = self.compute_rotation(kwargs['anuglar_velocity'])

        self._pose += np.array([
            delta_path * np.cos(self._pose[2] + delta_beta / 2),
            delta_path * np.sin(self._pose[2] + delta_beta / 2), delta_beta
        ])

        self._pose = np.round(self._pose, 3)
        self._pose[2] = self._angle_correction(self._pose[2])

        self._polar_coordinates[0] = np.sqrt(
            np.sum((self._target_position - self._pose[0:2]) ** 2))

        theta = np.arctan2(self._target_position[1] - self._pose[1],
                           self._target_position[0] - self._pose[0])

        theta = self._angle_correction(theta)

        self._polar_coordinates[1] = self._angle_correction(
            theta - self._pose[2])

    def compute_delta_motion(self, phi):
        wheels_paths = phi * self._wheel_radius

        delta_path = np.round(np.sum(wheels_paths) / 2, 3)
        self._sum_path += delta_path

        return delta_path

    def compute_rotation(self, current_angular_velocity):
        delta_beta = np.round(current_angular_velocity * self._dt, 3)
        self._previous_angular_velocity = current_angular_velocity

        return delta_beta

    def reset(self, start_pose):
        self._pose = start_pose
        self._polar_coordinates = np.zeros(2)

        self._sum_path = 0.0
        self._previous_angular_velocity = 0.0
