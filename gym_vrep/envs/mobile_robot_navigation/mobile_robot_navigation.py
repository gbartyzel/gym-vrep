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


def _choose_world_type(enable_vision):
    if enable_vision:
        return 'mobile_robot_navigation_room_vision'
    return 'mobile_robot_navigation_room'


class MobileRobotNavigationEnv(gym_vrep.VrepEnv):
    metadata = {'render.modes': ['human']}
    navigation_type = NAVIGATION_TYPE['Ideal']
    enable_vision = False

    def __init__(self, dt):
        super(MobileRobotNavigationEnv, self).__init__(
            _choose_world_type(self.enable_vision), dt)
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

        self._goal = np.array([2.0, 2.0])

        self._prev_polar_coords = np.zeros(2)

        self._alpha_factor = 25
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

        if self.enable_vision:
            self.observation_space = spaces.Dict(dict(
                image=spaces.Box(
                    low=0, high=255, shape=(640, 480, 3), dtype=np.uint8),
                scalars=spaces.Box(
                    -np.inf, np.inf, shape=(2, ), dtype=np.float32),
            ))
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(7, ), dtype=np.float32)

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
        if self.enable_vision:
            image = self._robot.get_image()
            state = {'image': image, 'scalars': polar_coords}

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
        if self.enable_vision:
            image = self._robot.get_image()
            state = {'image': image, 'scalars': polar_coords}

        return state


class MobileRobotOdomNavigationEnv(MobileRobotNavigationEnv):
    navigation_type = NAVIGATION_TYPE['Odometry']


class MobileRobotGyroNavigationEnv(MobileRobotNavigationEnv):
    navigation_type = NAVIGATION_TYPE['Gyrodometry']


class MobileRobotVisionNavigationEnv(MobileRobotNavigationEnv):
    enable_vision = True
