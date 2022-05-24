from typing import Dict
from typing import Tuple
from typing import Union

import numpy as np
from gym import spaces
from pyrep.backend import sim
from pyrep.objects.dummy import Dummy

from gym_coppelia_sim.envs import gym_coppelia_sim
from gym_coppelia_sim.envs.mobile_robot_navigation.navigation_algos import Gyrodometry
from gym_coppelia_sim.envs.mobile_robot_navigation.navigation_algos import Ideal
from gym_coppelia_sim.envs.mobile_robot_navigation.navigation_algos import Odometry
from gym_coppelia_sim.envs.mobile_robot_navigation.robots import PioneerP3Dx
from gym_coppelia_sim.envs.mobile_robot_navigation.robots import SmartBot

NAVIGATION_TYPE = {
    "Ideal": Ideal,
    "Odometry": Odometry,
    "Gyrodometry": Gyrodometry,
}

SPAWN_LIST = np.array([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])

GOAL_LIST = np.array([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])

ENV_TUPLE = Tuple[
    Union[Dict[str, np.ndarray], np.ndarray], float, bool, Dict[str, bool]
]

ENV_TUPLE_WO_STATE = Tuple[float, bool, Dict[str, bool]]


class NavigationEnv(gym_coppelia_sim.CoppeliaSimEnv):
    """The gym environment for mobile robot navigation task.

    Six variants of this environment are given:
    * Ideal: a position of mobile robot is received from simulation engine,
    * Odometry: a position o mobile robot is computed by encoders ticks,
    * Gyrodometry: a position of mobile robot is computed by encoders ticks and
    readings from gyroscope
    * Vision: it's a ideal variant with camera in stace space instead of
    proximity sensors reading.
    * Dynamic:
    * DynamicVision

    The state space of this environment includes proximity sensors readings
    (or image from camera), polar coordinates of mobile robot, linear and
    angular velocities. Action space are target motors velocities in rad/s.
    The reward function is based on robot velocity, heading angle and
    distance from nearest obstacle.

    """

    metadata = {"render.modes": ["human"]}
    navigation_type = NAVIGATION_TYPE["Ideal"]
    enable_vision = False

    def __init__(self, scene: str = "room.ttt", dt: float = 0.05):
        """A class constructor.

        Args:
            dt: Delta time of simulation.
        """
        super(NavigationEnv, self).__init__(scene=scene, dt=dt, headless_mode=False)

        self._goal_threshold = 0.05
        self._collision_dist = 0.05

        self._robot = SmartBot(enable_vision=self.enable_vision)
        self._obstacles = sim.simGetObjectHandle("Obstacles_visual")
        self._navigation = self.navigation_type(self._robot, dt)

        self._goal = Dummy.create(size=0.1)
        self._goal.set_renderable(True)
        self._goal.set_name("Goal")

        max_linear_vel = (
            self._robot.wheel_radius * 2 * self._robot.velocity_limit[1] / 2
        )
        max_angular_vel = (
            self._robot.wheel_radius
            / self._robot.wheel_distance
            * np.diff(self._robot.velocity_limit)
        )[0]

        self.action_space = spaces.Box(
            *self._robot.velocity_limit,
            shape=self._robot.velocity_limit.shape,
            dtype="float32"
        )

        low = self._get_lower_observation(max_angular_vel)
        high = self._get_upper_observation(max_linear_vel, max_angular_vel)

        if self.enable_vision:
            self.observation_space = spaces.Dict(
                dict(
                    image=spaces.Box(
                        low=0, high=255, shape=self._robot.image.shape, dtype=np.uint8
                    ),
                    scalars=spaces.Box(low=low, high=high, dtype=np.float32),
                )
            )
        else:
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action: np.ndarray) -> ENV_TUPLE:
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

        self._robot.set_joint_target_velocities(list(action))
        self._pr.step()
        sclaras = self._get_scalar_observation()
        reward, done, info = self._compute_reward(sclaras)

        state = sclaras
        if self.enable_vision:
            state = {
                "scalars": sclaras,
                "image": (self._robot.image * 256.0).astype(np.uint8),
            }
        return state, reward, done, info

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Resets environment to initial state.

        Returns:
            state: The sensors readings, polar coordinates, linear and
            angular velocities.
        """

        self._pr.set_configuration_tree(self._robot.initial_configuration)
        self._robot.set_motor_locked_at_zero_velocity(True)
        self._robot.set_control_loop_enabled(False)

        start_pose = self._sample_start_parameters()
        self._robot.set_2d_pose(list(start_pose))
        self._randomize_object_color(self._obstacles)

        self._navigation.reset(start_pose)
        self._pr.step()
        if self.enable_vision:
            return {
                "scalars": self._get_scalar_observation(),
                "image": (self._robot.image * 256.0).astype(np.uint8),
            }

        return self._get_scalar_observation()

    def _compute_reward(self, state: np.ndarray) -> ENV_TUPLE_WO_STATE:
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
        info = {"is_success": False}
        offset = self._robot.nb_ultrasonic_sensor

        reward = state[offset + 2] * np.cos(state[offset + 1])

        if (state[0:offset] < 0.1).any():
            reward = -0.1

        if (state[0:offset] < self._collision_dist).any():
            reward = -1.0
            done = True

        if state[offset + 0] <= self._goal_threshold:
            reward = 1.0
            info = {"is_success": True}
            done = True

        return reward, done, info

    def _get_scalar_observation(self) -> np.ndarray:
        """Gets current observation space from environment.

        Returns:
            state: The sensors readings, polar coordinates, linear and angular
            velocities.

        """
        ultrasonic_distance = self._robot.ultrasonic_distances
        polar_coords = self._navigation.compute_position(
            np.asarray(self._goal.get_position()[0:2])
        )
        velocities = self._robot.get_base_velocities()
        state = np.concatenate((ultrasonic_distance, polar_coords, velocities))
        return state

    def _get_lower_observation(self, max_angular_vel: float) -> np.ndarray:
        """Gets lowest values of observation space.

        Args:
            max_angular_vel: Maximum angular velocity of mobile robot

        Returns:
            low_boundaries: Lowest values of observation space.

        """
        ultrasonic_distance = (
            np.ones(self._robot.nb_ultrasonic_sensor)
            * self._robot.ultrasonic_sensor_bound[0]
        )
        polar_coords = np.array([0.0, -np.pi])
        velocities = np.array([0.0, -max_angular_vel])

        return np.concatenate((ultrasonic_distance, polar_coords, velocities))

    def _get_upper_observation(
        self, max_linear_vel: float, max_angular_vel: float
    ) -> np.ndarray:
        """Gets highest values of observation space

        Args:
            max_linear_vel: Maximum linear velocity of mobile robot.
            max_angular_vel: Maximum angular velocity of mobile robot.

        Returns:
            high_boundaries: Highest values of observation space.

        """
        env_diagonal = np.sqrt(2.0 * (5.0**2))
        ultrasonic_distance = (
            np.ones(self._robot.nb_ultrasonic_sensor)
            * self._robot.ultrasonic_sensor_bound[1]
        )
        polar_coords = np.array([env_diagonal, np.pi])
        velocities = np.array([max_linear_vel, max_angular_vel])

        return np.concatenate((ultrasonic_distance, polar_coords, velocities))

    def _sample_start_parameters(self) -> np.ndarray:
        """A method that generates mobile robot start pose and desired goal/

        Returns:
            goal: Desired position of mobile robot.
            start_pose: Start pose of mobile robot
        """
        idx = np.random.randint(GOAL_LIST.shape[0])
        goal = self._generate_goal(idx)
        start_pose = self._generate_start_pose(idx)
        self._goal.set_position(
            list(goal)
            + [
                0.0,
            ]
        )
        return start_pose

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
        position += np.random.uniform(-0.1, 0.1, (2,))
        orientation = np.rad2deg(np.random.uniform(-np.pi, np.pi))

        pose = np.concatenate((position, np.array([orientation])))
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

    @staticmethod
    def _randomize_object_color(object_handle: int):
        color = list(np.random.uniform(low=0.0, high=1.0, size=(3,)))
        sim.simSetShapeColor(
            object_handle, None, sim.sim_colorcomponent_ambient_diffuse, color
        )


class OdomNavigationEnv(NavigationEnv):
    """Odometry variant of environment."""

    navigation_type = NAVIGATION_TYPE["Odometry"]


class GyroNavigationEnv(NavigationEnv):
    """Gyrodometry variant of environment."""

    navigation_type = NAVIGATION_TYPE["Gyrodometry"]


class VisionNavigationEnv(NavigationEnv):
    """Vision feedback variant of environment."""

    enable_vision = True


class DynamicNavigationEnv(NavigationEnv):
    """Environment variant with moving robots as dynamic obstacles.In this
    environment robot has additional ultrasonic sensors in the back.
    """

    def __init__(self, dt: float = 0.05):
        SmartBot.nb_ultrasonic_sensor = 10
        super(DynamicNavigationEnv, self).__init__("dynamic_room.ttt", dt)
        self._dummy_robots = [PioneerP3Dx(count=i) for i in range(4)]
        self._initials_robots_config = [
            robot.get_configuration_tree() for robot in self._dummy_robots
        ]

    def reset(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        for conf in self._initials_robots_config:
            self._pr.set_configuration_tree(conf)
        for robot in self._dummy_robots:
            pose = robot.get_2d_pose()
            pose[2] += np.random.uniform(-np.pi / 2.0, np.pi / 2.0)
            robot.set_2d_pose(pose)
        return super().reset()


class DynamicVisionNavigationEnv(DynamicNavigationEnv):
    """Environment variant with moving robots as dynamic obstacles.In this
    environment robot has additional ultrasonic sensors in the back and also
    camera.
    """

    enable_vision = True
