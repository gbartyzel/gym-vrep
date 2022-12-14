from typing import Optional, Tuple

import numpy as np
from gym import spaces
from pyrep.backend import sim
from pyrep.objects.dummy import Dummy

from gym_coppelia_sim.common.typing import ArrayStruct, EnvironmentTuple
from gym_coppelia_sim.envs import gym_coppelia_sim
from gym_coppelia_sim.envs.mobile_robot_navigation.navigation_algos import (
    NavigationAlgorithm,
)
from gym_coppelia_sim.robots import PioneerP3Dx, SmartBot


class NavigationEnv(gym_coppelia_sim.CoppeliaSimEnv):
    """The gym environment for mobile robot navigation task.

    The environment can be configured with following parameters:
    * `navigation_type` - defines which nagivation algorithm is used by in the
    environment. Currently supported algorithms:
      * `ideal` - a position of mobile robot is received from simulation engine.
      * `odometry` - a position of mobile robot is computed by encoders ticks and
      readings from gyroscope.
      * `gyrodometry` - a position o mobile robot is computed by encoders ticks.
    * `enable_vision` - whether to use visual feedback in the observation space.
    * `goal_threshold` - defines the precision of the navgiation to the given goal.
    * `collision_threshold` - defines when the collision occurs.

    The state space of this environment includes proximity sensors readings
    (or image from camera), polar coordinates of mobile robot, linear and
    angular velocities. Action space are target motors velocities in rad/s.
    The reward function is based on robot velocity, heading angle and
    distance from nearest obstacle.
    """

    spawn_positions = np.array([[-2.0, -2.0], [2.0, -2.0], [-2.0, 2.0], [2.0, 2.0]])
    goal_positions = np.array([[2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]])

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        scene: str = "room.ttt",
        dt: float = 0.05,
        headless_mode: bool = False,
        navigation_type: str = "ideal",
        enable_vision: bool = False,
        goal_threshold: float = 0.05,
        collision_threshold: float = 0.05,
    ):
        """
        Initialize class object.

        Args:
            scene: Name of the scene to be loaded.
            dt: A time step of the simulation.
            model: Name of the model that to be imported.
            headless_mode: Defines mode of the simulation.
            navigation_type: Type of the navigation algorithm used for the env.
            enable_vision: Whether to use visual observation or not.
            goal_threshold: Defines the goal threshold.
            collision_threshold: Defines the collision threshold.
        Args:
            dt: Delta time of simulation.
        """
        super().__init__(scene=scene, dt=dt, headless_mode=headless_mode)

        self._goal_threshold = goal_threshold
        self._collision_threshold = collision_threshold
        self._enable_vision = enable_vision

        self._robot = SmartBot(enable_vision=enable_vision)
        self._obstacles = sim.simGetObjectHandle("Obstacles_visual")
        self._navigation = NavigationAlgorithm.build(
            algo_type=navigation_type, robot=self._robot, dt=dt
        )

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
            dtype=np.float32,
        )

        low = self._get_lower_observation(max_angular_vel)
        high = self._get_upper_observation(max_linear_vel, max_angular_vel)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        if self._enable_vision:
            self.observation_space = spaces.Dict(
                dict(
                    image=spaces.Box(
                        low=0, high=255, shape=self._robot.image.shape, dtype=np.uint8
                    ),
                    scalars=self.observation_space,
                )
            )

    def step(self, action: np.ndarray) -> EnvironmentTuple:
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
        if self._enable_vision:
            state = {
                "scalars": sclaras,
                "image": (self._robot.image * 256.0).astype(np.uint8),
            }
        return state, reward, done, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ArrayStruct:
        """Resets environment to initial state.

        Returns:
            state: The sensors readings, polar coordinates, linear and
            angular velocities.
        """
        super().reset(seed=seed, return_info=return_info, options=options)

        self._pr.set_configuration_tree(self._robot.initial_configuration)
        self._robot.set_motor_locked_at_zero_velocity(True)
        self._robot.set_control_loop_enabled(False)

        start_pose = self._sample_start_parameters()
        self._robot.set_2d_pose(list(start_pose))
        self._randomize_object_color(self._obstacles)

        self._navigation.reset(start_pose)
        self._pr.step()
        if self._enable_vision:
            return {
                "scalars": self._get_scalar_observation(),
                "image": (self._robot.image * 256.0).astype(np.uint8),
            }

        return self._get_scalar_observation()

    def _compute_reward(self, state: np.ndarray) -> Tuple[float, bool, dict]:
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

        if (state[0:offset] < self._collision_threshold).any():
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
        idx = np.random.randint(self.goal_positions.shape[0])
        goal = self._generate_goal(idx)
        start_pose = self._generate_start_pose(idx)
        self._goal.set_position(goal.tolist() + [0.0])
        return start_pose

    def _generate_start_pose(self, idx: int) -> np.ndarray:
        """A method that generates mobile robot start pose. Pose is chosen from
        four variants.
        To chosen pose uniform noise is applied.

        Args:
            idx: The sampled index,

        Returns:
            pose: Generated start pose of mobile robot
        """
        position = np.take(self.spawn_positions, idx, axis=0)
        position += np.random.uniform(-0.1, 0.1, (2,))
        orientation = np.rad2deg(np.random.uniform(-np.pi, np.pi))

        pose = np.concatenate((position, np.array([orientation])))
        return pose

    def _generate_goal(self, idx: int) -> np.ndarray:
        """A method that generates goal position for mobile robot. Desired
        position is chosen from
        four variants. To chosen goal uniform noise is applied.

        Args:
            idx: The sampled index,

        Returns:
            goal: Generated goal.
        """
        goal = np.take(self.goal_positions, idx, axis=0)
        goal += np.random.uniform(-0.1, 0.1, (2,))
        return np.round(goal, 2)

    @staticmethod
    def _randomize_object_color(object_handle: int):
        color = list(np.random.uniform(low=0.0, high=1.0, size=(3,)))
        sim.simSetShapeColor(
            object_handle, None, sim.sim_colorcomponent_ambient_diffuse, color
        )


class DynamicNavigationEnv(NavigationEnv):
    """Environment variant with moving robots as dynamic obstacles.In this
    environment robot has additional ultrasonic sensors in the back.
    """

    def __init__(self, **kwargs):
        kwargs["scene"] = "dynamic_room.ttt"
        SmartBot.nb_ultrasonic_sensor = 10
        super().__init__(**kwargs)

        self._dummy_robots = [PioneerP3Dx(count=i) for i in range(4)]
        self._initials_robots_config = [
            robot.get_configuration_tree() for robot in self._dummy_robots
        ]

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ArrayStruct:
        for conf in self._initials_robots_config:
            self._pr.set_configuration_tree(conf)
        for robot in self._dummy_robots:
            pose = robot.get_2d_pose()
            pose[2] += np.random.uniform(-np.pi / 2.0, np.pi / 2.0)
            robot.set_2d_pose(pose)
        return super().reset(seed=seed, return_info=return_info, options=options)
