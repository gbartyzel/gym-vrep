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


class VrepMobileRobotNavigationEnv(gym_vrep.VrepEnv):
    metadata = {'render.modes': ['human']}
    navigation_type = NAVIGATION_TYPE['Ideal']

    def __init__(self, dt):
        super(VrepMobileRobotNavigationEnv, self).__init__(
            'mobile_robot_navigation_room', dt)

        v_rep_obj_names = {
            'left_motor': 'smartBotLeftMotor',
            'right_motor': 'smartBotRightMotor',
            'robot': 'smartBot',
        }
        v_rep_stream_names = {
            'proximity_sensor': 'proximitySensorsSignal',
            'encoders': 'encodersSignal',
            'accelerometer': 'accelerometerSignal',
            'gyroscope': 'gyroscopeSignal',
        }

        self._goal = np.array([2.0, 2.0])

        self._prev_polar_coords = np.zeros(2)

        self._alpha_factor = 25
        self._goal_threshold = 0.02
        self._collision_dist = 0.04
        self._env_steps = 0

        self._robot = Robot(self._client, self._dt, v_rep_obj_names,
                            v_rep_stream_names)

        self._navigation = self.navigation_type(
            self._goal, self._robot.wheel_diameter, self._robot.body_width,
            self._dt)

        self.action_space = spaces.Box(
            self._robot.velocity_bound[0],
            self._robot.velocity_bound[1],
            shape=self._robot.velocity_bound.shape,
            dtype='float32')

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(7, ), dtype='float32')

    def step(self, action):
        self._robot.set_motor_velocities(action)
        vrep.simxSynchronousTrigger(self._client)
        vrep.simxGetPingTime(self._client)

        cart_pose = self._robot.get_position()
        gyro_data = self._robot.get_gyroscope_values()
        delta_phi = self._robot.get_encoders_rotations()

        self._navigation.compute_position(
            position=cart_pose, phi=delta_phi, anuglar_velocity=gyro_data[2])

        polar_coords = self._navigation.polar_coordinates
        prox_dist = self._robot.get_proximity_values()
        done = False

        state = np.concatenate((prox_dist, polar_coords))

        ds = self._prev_polar_coords[0] - polar_coords[0]
        reward = ds * self._alpha_factor

        if not np.all(prox_dist > self._collision_dist):
            reward = -1.0
            done = True

        if polar_coords[0] < self._goal_threshold:
            reward = 1.0
            done = True

        if done:
            vrep.simxStopSimulation(self._client,
                                    vrep.simx_opmode_oneshot_wait)

        self._prev_polar_coords = polar_coords

        return state, reward, done, {}

    def reset(self):
        vrep.simxStopSimulation(self._client, vrep.simx_opmode_oneshot_wait)
        self._robot.reset()
        vrep.simxStartSimulation(self._client, vrep.simx_opmode_oneshot_wait)

        self._env_steps = 0
        for _ in range(2):
            vrep.simxSynchronousTrigger(self._client)
            vrep.simxGetPingTime(self._client)

        cart_pose = self._robot.get_position()
        self._navigation.reset(cart_pose)

        prox_dist = self._robot.get_proximity_values()
        gyro_data = self._robot.get_gyroscope_values()
        delta_phi = self._robot.get_encoders_rotations()

        self._navigation.compute_position(
            position=cart_pose, phi=delta_phi, anuglar_velocity=gyro_data[2])

        polar_coords = self._navigation.polar_coordinates
        self._prev_polar_coords = polar_coords

        state = np.concatenate((prox_dist, polar_coords))

        return state


class VrepMobileRobotOdomNavigationEnv(VrepMobileRobotNavigationEnv):
    navigation_type = NAVIGATION_TYPE['Odometry']


class VrepMobileRobotGyroNavigationEnv(VrepMobileRobotNavigationEnv):
    navigation_type = NAVIGATION_TYPE['Gyrodometry']


class VrepMobileRobotNavigationGoalEnv(VrepMobileRobotNavigationEnv):
    metadata = {'render.modes': ['human']}
    navigation_type = NAVIGATION_TYPE['Ideal']

    goal_env = False

    def __init__(self, dt):
        super(VrepMobileRobotNavigationEnv, self).__init__(
            'mobile_robot_navigation_room', dt)

        v_rep_obj_names = {
            'left_motor': 'smartBotLeftMotor',
            'right_motor': 'smartBotRightMotor',
            'robot': 'smartBot',
        }
        v_rep_stream_names = {
            'proximity_sensor': 'proximitySensorsSignal',
            'encoders': 'encodersSignal',
            'accelerometer': 'accelerometerSignal',
            'gyroscope': 'gyroscopeSignal',
        }

        self._goal = self._goal_generator()

        self._prev_polar_coords = np.zeros(2)

        self._goal_threshold = 0.02
        self._collision_dist = 0.04

        self._robot = Robot(self._client, self._dt, v_rep_obj_names,
                            v_rep_stream_names)

        self._navigation = self.navigation_type(
            self._goal, self._robot.wheel_diameter, self._robot.body_width,
            self._dt)

        self.action_space = spaces.Box(
            self._robot.velocity_bound[0],
            self._robot.velocity_bound[1],
            shape=self._robot.velocity_bound.shape,
            dtype='float32')

        self.observation_space = spaces.Dict(
            dict(
                desired_goal=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=self._navigation.pose[:2].shape,
                    dtype='float32'),
                achieved_goal=spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=self._navigation.pose[:2].shape,
                    dtype='float32'),
                observation=spaces.Box(
                    -np.inf, np.inf, shape=(7, ), dtype='float32')))

    def step(self, action):
        self._robot.set_motor_velocities(action)
        vrep.simxSynchronousTrigger(self._client)
        vrep.simxGetPingTime(self._client)

        cart_pose = self._robot.get_position()
        gyro_data = self._robot.get_gyroscope_values()
        delta_phi = self._robot.get_encoders_rotations()

        self._navigation.compute_position(
            position=cart_pose, phi=delta_phi, anuglar_velocity=gyro_data[2])

        polar_coords = self._navigation.polar_coordinates
        prox_dist = self._robot.get_proximity_values()
        done = False
        info = {
            'is_success': 0.0,
        }

        state = {
            'observation':
            np.concatenate((prox_dist, self._navigation.pose[:2])).copy(),
            'desired_goal':
            self._goal.copy(),
            'achieved_goal':
            self._navigation.pose[:2].copy(),
        }
        reward = -1.0

        if not np.all(prox_dist > self._collision_dist):
            reward = -1.0
            done = True

        if polar_coords[0] < self._goal_threshold:
            reward = 1.0
            info['is_success'] = 1.0
            done = True

        if done:
            vrep.simxStopSimulation(self._client,
                                    vrep.simx_opmode_oneshot_wait)

        self._prev_polar_coords = polar_coords

        return state, reward, done, info

    def reset(self):
        vrep.simxStopSimulation(self._client, vrep.simx_opmode_oneshot_wait)
        self._robot.reset()
        vrep.simxStartSimulation(self._client, vrep.simx_opmode_oneshot_wait)

        self._goal = self._goal_generator()

        for _ in range(2):
            vrep.simxSynchronousTrigger(self._client)
            vrep.simxGetPingTime(self._client)

        cart_pose = self._robot.get_position()
        self._navigation.reset(cart_pose)

        prox_dist = self._robot.get_proximity_values()
        gyro_data = self._robot.get_gyroscope_values()
        delta_phi = self._robot.get_encoders_rotations()

        self._navigation.compute_position(
            position=cart_pose, phi=delta_phi, anuglar_velocity=gyro_data[2])

        polar_coords = self._navigation.polar_coordinates
        self._prev_polar_coords = polar_coords

        state = {
            'observation': np.concatenate((prox_dist,
                                           self._navigation.pose[:2])),
            'desired_goal': self._goal,
            'achieved_goal': self._navigation.pose[:2],
        }

        return state

    def compute_reward(self, achieved_goal, desired_goal, info):
        d = np.linalg.norm(achieved_goal - desired_goal)
        if d < self._goal_threshold:
            return 0
        return -1

    def _goal_generator(self):
        goals = [
            np.array([-2.0, 2.0]) + self.np_random.uniform(-0.1, 0.1, 2),
            np.array([2.0, 2.0]) + self.np_random.uniform(-0.1, 0.1, 2),
            np.array([2.0, -2.0]) + self.np_random.uniform(-0.1, 0.1, 2)
        ]
        idx = self.np_random.randint(0, 3)
        return np.round(goals[idx], 3)
