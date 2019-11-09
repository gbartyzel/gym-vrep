from typing import Dict
from typing import Tuple

import numpy as np
from gym import spaces

import gym_vrep.envs.vrep.vrep_lib as vlib
from gym_vrep.envs import gym_vrep
from gym_vrep.envs.mobile_robot_navigation.navigation import Gyrodometry
from gym_vrep.envs.mobile_robot_navigation.navigation import Ideal
from gym_vrep.envs.mobile_robot_navigation.navigation import Odometry
from gym_vrep.envs.mobile_robot_navigation.robot import Robot

NAVIGATION_TYPE = {
    'Ideal': Ideal,
    'Odometry': Odometry,
    'Gyrodometry': Gyrodometry,
}

SPAWN_LIST = np.array([
    [-2.0, -2.0, 0.0405],
    [2.0, -2.0, 0.0405],
    [-2.0, 2.0, 0.0405],
    [2.0, 2.0, 0.0405]
])

GOAL_LIST = np.array([
    [2.0, 2.0],
    [-2.0, 2.0],
    [2.0, -2.0],
    [-2.0, -2.0]
])


def _choose_model(enable_vision: bool) -> str:
    """Switch between model with and without camera

    Args:
        enable_vision: Flag that enable model with camera

    Returns: Name of the model to be loaded

    """
    if enable_vision:
        return 'mobile_robot_with_camera'
    return 'mobile_robot'


class MobileRobotNavigationEnv(gym_vrep.VrepEnv):
    """The gym environment for mobile robot navigation task.

    Four variants of this environment are given:
    * Ideal: a position of mobile robot is received from simulation engine,
    * Odometry: a position o mobile robot is computed by encoders ticks,
    * Gyrodometry: a position of mobile robot is computed by encoders ticks and
    readings from gyroscope
    * Visual: it's a ideal variant with camera in stace space instead of
    proximity sensors reading.

    The state space of this environment includes proximity sensors readings
    (or image from camera), polar coordinates of mobile robot, linear and
    angular velocities. Action space are target motors velocities in rad/s.
    The reward function is based on robot velocity, heading angle and
    distance from nearest obstacle.

    """
    metadata = {'render.modes': ['human']}
    navigation_type = NAVIGATION_TYPE['Ideal']
    enable_vision = False

    def __init__(self, dt: float):
        """A class constructor.

        Args:
            dt: Delta time of simulation.
        """
        scene = 'mobile_robot_navigation_room'
        super(MobileRobotNavigationEnv, self).__init__(
            scene=scene,
            model=_choose_model(self.enable_vision),
            dt=dt)

        self._goal = None
        self._goal_threshold = 0.05
        self._collision_dist = 0.05
        self._time_length = 3

        self._robot = Robot(self._client, self._dt, self.enable_vision)
        self._navigation = self.navigation_type(self._robot, dt)

        radius = self._robot.wheel_diameter / 2.0
        max_linear_vel = radius * 2 * self._robot.velocity_limit[1] / 2
        max_angular_vel = radius / self._robot.body_width * np.diff(
            self._robot.velocity_limit)

        self.action_space = spaces.Box(*self._robot.velocity_limit,
                                       shape=self._robot.velocity_limit.shape,
                                       dtype='float32')

        low = self._get_observation_low(max_angular_vel)
        high = self._get_observation_high(max_linear_vel, max_angular_vel)

        if self.enable_vision:
            self.observation_space = spaces.Dict(dict(
                image=spaces.Box(
                    low=0, high=255, shape=(640, 480, 3), dtype=np.uint8),
                scalars=spaces.Box(low=low, high=high, dtype=np.float32),
            ))
        else:
            self.observation_space = spaces.Box(
                low=low, high=high, dtype=np.float32)

        self._state = np.zeros(low.shape)

    def step(self, action: np.ndarray) -> Tuple[
        np.ndarray, float, bool, Dict[str, bool]]:
        """Performs simulation step by applying given action.

        Args:
            action: A desired motors velocities.

        Returns:
            state: The sensors readings, polar coordinates, linear and
            angular velocities.
            reward: The reward received in current simulation step for
            state-action pair
            done: Flag if environment is finished.
            info: Dictionary containing diagnostic information.

        Raises:
            ValueError: Dimension of input motors velocities is incorrect!
        """
        self._robot.set_motor_velocities(action)
        vlib.trigger_simulation_step(self._client)

        state = self._get_observation()

        reward, done, info = self._compute_reward(
            state['scalars'] if self.enable_vision else state)

        self._state = np.roll(self._state, -1, 0)
        if isinstance(state, dict):
            self._state[0] = state['scalars']
        else:
            self._state[0] = state

        return self._state, reward, done, info

    def reset(self) -> np.ndarray:
        """Resets environment to initial state.

        Returns:
            state: The sensors readings, polar coordinates, linear and
            angular velocities.
        """
        self._goal, start_pose = self._sample_start_parameters()
        print('Start position: {}, Current goal: {}'.format(
            np.round(start_pose[0:2], 2), self._goal))
        vlib.stop_simulation(self._client)
        self._start()

        self._set_start_pose(self._robot.robot_handle, start_pose)

        self._navigation.reset(np.append(start_pose[0:3], start_pose[3:]))
        state = self._get_observation()
        vlib.trigger_simulation_step(self._client)

        if isinstance(state, dict):
            self._state = np.stack((state['scalars'], ) * self._time_length)
        else:
            self._state = np.stack((state, ) * self._time_length)
        return self._state

    def _compute_reward(self, state: np.ndarray) -> Tuple[
        float, bool, Dict[str, bool]]:
        """Computes reward for current state-action pair.

        Args:
            state: The sensors readings, polar coordinates, linear and angular
            velocities.

        Returns:
            reward: The reward received in current simulation step for
            state-action pair
            done: Flag if environment is finished.
            info: Dictionary that contain information if robot successfully
            finished task.

        """
        done = False
        info = {'is_success': False}

        reward_navigation = state[7] * np.cos(state[6])
        reward = reward_navigation

        if (state[0:5] < 0.1).any():
            reward = -0.1

        if (state[0:5] < self._collision_dist).any():
            reward = -1.0
            done = True

        if state[5] <= self._goal_threshold:
            reward = 1.0
            info = {'is_success': True}
            done = True

        return reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Gets current observation space from environment.

        Returns:
            state: The sensors readings, polar coordinates, linear and angular
            velocities.

        """
        proximity_sensor_distance = self._robot.get_proximity_values()
        polar_coordinates = self._navigation.compute_position(self._goal)
        velocities = self._robot.get_velocities()

        state = np.concatenate(
            (proximity_sensor_distance, polar_coordinates, velocities))

        if self.enable_vision:
            image = self._robot.get_image()
            state = {'image': image, 'scalars': state}

        return state

    def _get_observation_low(self, max_angular_vel: float) -> np.ndarray:
        """Gets lowest values of observation space.

        Args:
            max_angular_vel: Maximum angular velocity of mobile robot

        Returns:
            low_boundaries: Lowest values of observation space.

        """
        proximity_sensor = (
                np.ones(self._robot.nb_proximity_sensor) *
                self._robot.ultrasonic_sensor_bound[0])
        polar_coordinates = np.array([0.0, -np.pi])
        velocities = np.array([0.0, -max_angular_vel])

        low_boundaries = np.concatenate(
            (proximity_sensor, polar_coordinates, velocities))
        return np.stack((low_boundaries, ) * self._time_length)

    def _get_observation_high(self, max_linear_vel: float,
                              max_angular_vel: float) -> np.ndarray:
        """Gets highest values of observation space

        Args:
            max_linear_vel: Maximum linear velocity of mobile robot.
            max_angular_vel: Maximum angular velocity of mobile robot.

        Returns:
            high_boundaries: Highest values of observation space.

        """
        env_diagonal = np.sqrt(2.0 * (5.0 ** 2))
        proximity_sensor = (
                np.ones(self._robot.nb_proximity_sensor) *
                self._robot.ultrasonic_sensor_bound[1])
        polar_coordinates = np.array([env_diagonal, np.pi])
        velocities = np.array([max_linear_vel, max_angular_vel])

        high_boundaries = np.concatenate(
            (proximity_sensor, polar_coordinates, velocities))
        return np.stack((high_boundaries, ) * self._time_length)

    def _sample_start_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """A method that generates mobile robot start pose and desired goal/

        Returns:
            goal: Desired position of mobile robot.
            start_pose: Start pose of mobile robot
        """
        idx = np.random.randint(GOAL_LIST.shape[0])
        goal = self._generate_goal(idx)
        start_pose = self._generate_start_pose(idx)

        return goal, start_pose

    @staticmethod
    def _generate_start_pose(idx: int) -> np.ndarray:
        """A method that generates mobile robot start pose. Pose is chosen from
        four variants.
        To chosen pose uniform noise is applied.

        Args:
            idx: The sampled index,

        Returns:
            pose: Generated start pose of mobile robot
        """
        position = np.take(SPAWN_LIST, idx, axis=0)
        position[0:2] += np.random.uniform(-0.1, 0.1, (2,))
        orientation = np.zeros(3)
        orientation[2] = np.rad2deg(np.random.uniform(-np.pi, np.pi))

        pose = np.concatenate((position, orientation))
        return pose

    @staticmethod
    def _generate_goal(idx: int) -> np.ndarray:
        """A method that generates goal position for mobile robot. Desired
        position is chosen from
        four variants. To chosen goal uniform noise is applied.

        Args:
            idx: The sampled index,

        Returns:
            goal: Generated goal.
        """
        goal = np.take(GOAL_LIST, idx, axis=0)
        noise = np.random.uniform(-0.1, 0.1, (2,))

        goal += noise
        return np.round(goal, 2)


class MobileRobotOdomNavigationEnv(MobileRobotNavigationEnv):
    """Odometry variant of environment.

    """
    navigation_type = NAVIGATION_TYPE['Odometry']


class MobileRobotGyroNavigationEnv(MobileRobotNavigationEnv):
    """Gyrodometry variant of environment.

    """
    navigation_type = NAVIGATION_TYPE['Gyrodometry']


class MobileRobotVisionNavigationEnv(MobileRobotNavigationEnv):
    """Visual variant of environment.

    """
    enable_vision = True
