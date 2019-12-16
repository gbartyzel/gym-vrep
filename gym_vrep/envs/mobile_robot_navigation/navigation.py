import abc
from typing import Any
from typing import NoReturn
from typing import Tuple

import numpy as np

from gym_vrep.envs.mobile_robot_navigation.robot import SmartBot


class Base(metaclass=abc.ABCMeta):
    """A superclass for navigation tasks.
    """

    def __init__(self, robot: SmartBot, dt):
        """A constructor of superclass

        Args:
            robot: An object of mobile robot.
            dt: A delta time of simulation.
        """
        self._robot = robot
        self._target_position = None
        self._dt = dt

        self._pose = np.zeros(3)

    @abc.abstractmethod
    def compute_position(self, goal: np.ndarray) -> Any:
        """An abstract method for computing position.

        Args:
            goal: Desired goal of mobile robot.
        """
        return NotImplementedError

    @abc.abstractmethod
    def reset(self, start_pose: np.ndarray) -> Any:
        """An abstract method for object reset functionality.

        Args:
            start_pose: Start pose of mobile robot.
        """
        return NotImplementedError

    @staticmethod
    def _angle_correction(angle: float or np.ndarray) -> np.ndarray:
        """Static method that correct given angles into range -pi, pi

        Args:
            angle: An array of angles or single angle

        Returns:
            new_angle: Angles or angle after correction.
        """
        new_angle = None
        if angle >= 0:
            new_angle = np.fmod((angle + np.pi), (2 * np.pi)) - np.pi

        if angle < 0:
            new_angle = np.fmod((angle - np.pi), (2 * np.pi)) + np.pi

        new_angle = np.round(angle, 3)
        return new_angle


class Ideal(Base):
    """A class that computes robot polar coordinates of mobile robot based on
    absolute pose received from simulation engine and kinematic model.

    """

    def __init__(self, robot: SmartBot, dt: float):
        """A constructor of class.

        Args:
            robot: An object of mobile robot.
            dt: A delta time of simulation.
        """
        super(Ideal, self).__init__(robot, dt)

    def compute_position(self, goal) -> np.ndarray:
        """A method that computer polar coordinates of mobile robot.

        Args:
            goal: Desired goal of mobile robot.

        Returns:
            polar_coordinates: A polar coordinates of mobile robot.
        """
        pose = np.round(self._robot.get_2d_pose(), 3)
        pose[2] = self._angle_correction(pose[2])

        distance = np.linalg.norm(pose[0:2] - goal)

        theta = np.arctan2(goal[1] - pose[1], goal[0] - pose[0])

        theta = self._angle_correction(theta)

        heading_angle = self._angle_correction(theta - pose[2])

        polar_coordinates = np.round(np.array([distance, heading_angle]), 3)
        return polar_coordinates

    def reset(self, start_pose: np.ndarray) -> NoReturn:
        """A method that reset member variable.

        Args:
            start_pose: Start pose of mobile robot.
        """
        self._pose = start_pose


class Odometry(Base):
    """A class that computes robot polar coordinates based of mobile robot based
    on odometry and kinematic model.

    """

    def __init__(self, robot: SmartBot, dt: float):
        """A constructor of class.

        Args:
            robot: An object of mobile robot.
            dt: A delta time of simulation.
        """
        super(Odometry, self).__init__(robot, dt)

        self._sum_path = 0.0

    def compute_position(self, goal: np.ndarray) -> np.ndarray:
        """A method that computer polar coordinates of mobile robot.

        Args:
            goal: Desired goal of mobile robot.

        Returns:
            polar_coordinates: A polar coordinates of mobile robot.
        """
        delta_path, delta_beta = self.compute_delta_motion()

        self._pose += np.array([
            delta_path * np.cos(self._pose[2] + delta_beta / 2),
            delta_path * np.sin(self._pose[2] + delta_beta / 2), delta_beta
        ])

        self._pose = np.round(self._pose, 3)
        self._pose[2] = self._angle_correction(self._pose[2])

        distance = np.linalg.norm(self._pose[0:2] - goal)

        theta = np.arctan2(goal[1] - self._pose[1], goal[0] - self._pose[0])
        theta = self._angle_correction(theta)

        heading_angle = self._angle_correction(theta - self._pose[2])

        polar_coordinates = np.round(np.array([distance, heading_angle]), 3)

        return polar_coordinates

    def compute_delta_motion(self) -> Tuple[float, float]:
        """A method that compute difference between current traveled path and
        previous path. Also compute difference between current mobile robot
        rotation and previous one.

        Returns:
            delta_path: Delta of traveled path by robot
            delta_beta: Delta of rotation done by robot
        """
        wheels_paths = self._robot.encoder_ticks * self._robot.wheel_radius

        delta_path = np.round(np.sum(wheels_paths) / 2, 3)
        self._sum_path += delta_path

        delta_beta = (wheels_paths[1] - wheels_paths[0]) \
                     / self._robot.wheel_distance
        delta_beta = np.round(delta_beta, 3)

        return delta_path, delta_beta

    def reset(self, start_pose: np.ndarray) -> NoReturn:
        """A method that reset member variable.

        Args:
            start_pose: Start pose of mobile robot.
        """
        self._pose = start_pose
        self._sum_path = 0.0


class Gyrodometry(Odometry):
    """A class that computes robot polar coordinates based of mobile robot based
    on odometry, readings from gyroscope and kinematic model.

    """

    def __init__(self, robot: SmartBot, dt: float):
        """A constructor of class.

        Args:
            robot: An object of mobile robot.
            dt: A delta time of simulation.
        """
        super(Gyrodometry, self).__init__(robot, dt)

        self._previous_phi = 0.0
        self._previous_angular_velocity = 0.0

    def compute_position(self, goal: np.ndarray) -> np.ndarray:
        """A method that computer polar coordinates of mobile robot.

        Args:
            goal: Desired goal of mobile robot.

        Returns:
            polar_coordinates: A polar coordinates of mobile robot.
        """
        delta_path, _ = self.compute_delta_motion()
        delta_beta = self.compute_rotation()

        self._pose += np.array([
            delta_path * np.cos(self._pose[2] + delta_beta / 2),
            delta_path * np.sin(self._pose[2] + delta_beta / 2), delta_beta
        ])

        self._pose = np.round(self._pose, 3)
        self._pose[2] = self._angle_correction(self._pose[2])

        distance = np.linalg.norm(self._pose[0:2] - goal)

        theta = np.arctan2(goal[1] - self._pose[1], goal[0] - self._pose[0])
        theta = self._angle_correction(theta)
        heading_angle = self._angle_correction(theta - self._pose[2])

        return np.round(np.array([distance, heading_angle]), 3)

    def compute_rotation(self) -> float:
        """ A method that compute difference between current and previous
        rotation angle of mobile robot. The calculation are based on mobile
        robot angular velocity along z-axis.

        Returns:
            delta_beta: Delta of rotation done by robot

        """
        current_angular_velocity = self._robot.angular_velocities
        delta_beta = np.round(current_angular_velocity * self._dt, 3)
        self._previous_angular_velocity = current_angular_velocity

        return delta_beta

    def reset(self, start_pose: np.ndarray) -> NoReturn:
        """A method that reset member variable.

        Args:
            start_pose: Start pose of mobile robot.
        """
        self._pose = start_pose
        self._sum_path = 0.0
        self._previous_angular_velocity = 0.0
