import numpy as np

from gym import spaces
from gym_vrep.envs import gym_vrep
from gym_vrep.envs.vrep import vrep
from gym_vrep.envs.mobile_robot_navigation.robot import Robot
from gym_vrep.envs.mobile_robot_navigation.navigation import Ideal
from gym_vrep.envs.mobile_robot_navigation.navigation import Odometry
from gym_vrep.envs.mobile_robot_navigation.navigation import Gyrodometry

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


def _choose_model(enable_vision):
    if enable_vision:
        return 'mobile_robot_with_camera'
    return 'mobile_robot'


class MobileRobotNavigationEnv(gym_vrep.VrepEnv):
    metadata = {'render.modes': ['human']}
    navigation_type = NAVIGATION_TYPE['Ideal']
    enable_vision = False

    def __init__(self, dt):
        scene = 'mobile_robot_navigation_room'
        super(MobileRobotNavigationEnv, self).__init__(scene, _choose_model(self.enable_vision), dt)
        v_rep_obj_names = {
            'left_motor': 'smartBotLeftMotor',
            'right_motor': 'smartBotRightMotor',
            'robot': 'smartBot',
        }

        if self.enable_vision:
            v_rep_obj_names['camera'] = 'smartBotCamera'

        v_rep_stream_names = {
            'proximity_sensor': 'proximitySensorsSignal',
            'encoders': 'encodersSignal',
            'accelerometer': 'accelerometerSignal',
            'gyroscope': 'gyroscopeSignal',
        }

        self._goal = None

        self._goal_threshold = 0.05
        self._collision_dist = 0.05
        self._env_diagonal = np.sqrt(2.0 * (5.0 ** 2))
        self._prev_distance = 0.0

        self._robot = Robot(self._client, self._dt, v_rep_obj_names, v_rep_stream_names)
        self._navigation = self.navigation_type(
            self._robot.wheel_diameter, self._robot.body_width, self._dt)

        radius = self._robot.wheel_diameter / 2.0
        self._max_linear_vel = radius * 2 * self._robot.velocity_bound[1] / 2
        self._max_angular_vel = (
                radius / self._robot.body_width * np.diff(self._robot.velocity_bound))

        self.action_space = spaces.Box(self._robot.velocity_bound[0], self._robot.velocity_bound[1],
                                       shape=self._robot.velocity_bound.shape, dtype='float32')

        low = self._get_observation_low()
        high = self._get_observation_high()

        if self.enable_vision:
            self.observation_space = spaces.Dict(dict(
                image=spaces.Box(low=0, high=255, shape=(640, 480, 3), dtype=np.uint8),
                scalars=spaces.Box(low=low, high=high, dtype=np.float32),
            ))
        else:
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        self._robot.set_motor_velocities(action)
        vrep.simxSynchronousTrigger(self._client)

        state = self._get_observation()

        reward, done, info = self._compute_reward(state)

        return state, reward, done, info

    def reset(self):
        self._goal, start_pose = self._sample_start_parameters()
        print('Current goal: {}'.format(self._goal))

        vrep.simxStopSimulation(self._client, vrep.simx_opmode_blocking)
        self._spawn_robot(self._robot._object_names['robot'], start_pose)
        self._robot.reset()
        self._navigation.reset(start_pose, self._goal)
        vrep.simxStartSimulation(self._client, vrep.simx_opmode_blocking)

        for _ in range(2):
            vrep.simxSynchronousTrigger(self._client)
            vrep.simxGetPingTime(self._client)

        state = self._get_observation()
        self._prev_distance = state[5]

        return state

    def _compute_reward(self, state):
        done = False
        info = {'is_success': False}

        reward = state[7] * (-1.0) ** ((state[5] - self._prev_distance) > 0)

        if not np.all(state[0:5] > self._collision_dist):
            reward = -1.0
            done = True

        if state[5] <= self._goal_threshold:
            reward = 1.0
            info = {'is_success': True}
            done = True

        self._prev_distance = state[5]
        return reward, done, info

    def _get_observation(self):
        cartesian_pose = self._robot.get_position()
        proximity_sensor_distance = self._robot.get_proximity_values()
        gyroscope_angular_velocity = self._robot.get_gyroscope_values()
        delta_phi = self._robot.get_encoders_rotations()
        velocities = self._robot.get_velocities()

        self._navigation.compute_position(
            position=cartesian_pose, phi=delta_phi, anuglar_velocity=gyroscope_angular_velocity[2])
        polar_coordinates = self._navigation.polar_coordinates

        state = np.concatenate((proximity_sensor_distance, polar_coordinates, velocities))

        if self.enable_vision:
            image = self._robot.get_image()
            state = {'image': image, 'scalars': polar_coordinates}

        return state

    def _get_observation_low(self):
        proximity_sensor = (
                np.ones(self._robot.nb_proximity_sensor) * self._robot.proximity_sensor_bound[0])
        polar_coordinates = np.array([0.0, -np.pi])
        velocities = np.array([0.0, -self._max_angular_vel])

        return np.concatenate((proximity_sensor, polar_coordinates, velocities))

    def _get_observation_high(self):
        proximity_sensor = (
                np.ones(self._robot.nb_proximity_sensor) * self._robot.proximity_sensor_bound[1])
        polar_coordinatesh = np.array([self._env_diagonal, np.pi])
        velocities = np.array([self._max_linear_vel, self._max_angular_vel])

        return np.concatenate((proximity_sensor, polar_coordinatesh, velocities))

    def _sample_start_parameters(self):
        idx = np.random.randint(GOAL_LIST.shape[0])
        goal = self._generate_goal(idx)
        start_pose = self._generate_start_pose(idx)

        return goal, start_pose

    @staticmethod
    def _generate_start_pose(idx):
        position = np.take(SPAWN_LIST, idx, axis=0)
        position[0:2] += np.random.uniform(-0.1, 0.1, (2,))
        yaw_angle = np.rad2deg(np.random.uniform(-np.pi, np.pi))

        pose = {
            'position': np.round(position, 2),
            'orientation': np.array([0.0, 0.0, yaw_angle])
        }
        return pose

    @staticmethod
    def _generate_goal(idx):
        goal = np.take(GOAL_LIST, idx, axis=0)
        noise = np.random.uniform(-0.1, 0.1, (2,))
        return np.round(goal + noise, 2)


class MobileRobotOdomNavigationEnv(MobileRobotNavigationEnv):
    navigation_type = NAVIGATION_TYPE['Odometry']


class MobileRobotGyroNavigationEnv(MobileRobotNavigationEnv):
    navigation_type = NAVIGATION_TYPE['Gyrodometry']


class MobileRobotVisionNavigationEnv(MobileRobotNavigationEnv):
    enable_vision = True
