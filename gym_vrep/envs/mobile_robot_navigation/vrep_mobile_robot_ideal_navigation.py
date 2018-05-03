import numpy as np

from gym import spaces
from gym_vrep.envs import gym_vrep
from gym_vrep.envs.vrep import vrep
from gym_vrep.envs.mobile_robot_navigation.robot.base import Base
from gym_vrep.envs.mobile_robot_navigation.navigation.ideal import Ideal


class VrepMovileRobotIdealNavigationEnv(gym_vrep.VrepEnv):
    metadata = {'render.modes': ['human']}
    reward_range = (-1.0, 1.0)

    def __init__(self):
        super(VrepMovileRobotIdealNavigationEnv, self).__init__(
            'mobile_robot_navigation_room', 0.05)

        v_rep_obj_names = {
            'left_motor': 'smartBotLeftMotor',
            'right_motor': 'smartBotRightMotor',
            'robot': 'smartBot',
        }
        v_rep_stream_names = {
            'proximity_sensor': 'proximitySensorsSignal',
            'encoders': 'encodersSignal',
        }

        self._goal = np.array([2.0, 2.0])

        self._reward_params = {
            'penalty_dist': 0.04,
            'goal_precision': 0.01,
            'alpha_factor': 0.4,
        }

        self._robot = Base(self._client, self._dt, v_rep_obj_names,
                           v_rep_stream_names)

        self._navigation = Ideal(self._goal,
                                 self._robot.wheel_diameter,
                                 self._robot.body_width, self._dt)

        self.action_space = spaces.Box(self._robot.velocity_bound[0],
                                       self._robot.velocity_bound[1],
                                       (2,))

    def step(self, action):
        self._robot.set_motor_velocities(action)
        vrep.simxSynchronousTrigger(self._client)
        vrep.simxGetPingTime(self._client)

        cart_pose = self._robot.get_position()
        self._navigation.compute_position(position=cart_pose)
        polar_coords = self._navigation.polar_coordinates
        prox_dist= self._robot.get_proximity_values()

        state = np.concatenate((prox_dist, polar_coords))

        done = False
        reward = 0.0

        if not np.all(prox_dist > self._reward_params['penalty_dist']):
            reward = -1.0
            done = True

        if polar_coords[0] < self._reward_params['goal_precision']:
            reward = 1.0
            done = True

        if done:
            vrep.simxStopSimulation(self._client,
                                    vrep.simx_opmode_oneshot_wait)

        return state, reward, done, {}

    def reset(self):
        vrep.simxStopSimulation(self._client, vrep.simx_opmode_oneshot_wait)
        self._robot.reset()
        vrep.simxStartSimulation(self._client, vrep.simx_opmode_oneshot_wait)

        cart_pose = self._robot.get_position()

        self._navigation.reset(cart_pose)

        for _ in range(2):
            vrep.simxSynchronousTrigger(self._client)
            vrep.simxGetPingTime(self._client)

        prox_dist = self._robot.get_proximity_values()
        self._navigation.compute_position(position=cart_pose)
        polar_coords = self._navigation.polar_coordinates

        state = np.concatenate((prox_dist, polar_coords))

        return state
