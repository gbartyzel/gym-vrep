import os
import subprocess
import time

import gym
from gym.utils import seeding

from gym_vrep.envs.vrep import vrep


def run_env(port, scene):
    vrep_exec = os.environ['V_REP'] + 'vrep.sh -h '
    synch_mode_cmd = '-gREMOTEAPISERVERSERVICE_' + str(port) + '_FALSE_TRUE '
    fullpath = os.path.join(
        os.path.dirname(__file__), 'scenes', scene + '.ttt')

    subprocess.call(vrep_exec + synch_mode_cmd + fullpath + ' &', shell=True)
    time.sleep(0.5)

    client = vrep.simxStart('127.0.0.1', port, True, True, 5000, 5)
    vrep.simxSynchronous(client, True)

    return client


class VrepEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, scene, dt):
        self.seed(1337)

        self._dt = dt
        self._port = self.np_random.randint(20000, 21000)
        self._client = run_env(self._port, scene)
        vrep.simxSetFloatingParameter(self._client,
                                      vrep.sim_floatparam_simulation_time_step,
                                      dt, vrep.simx_opmode_blocking)
        print('Connected to port {}'.format(self._port))

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        subprocess.call('pkill -9 vrep &', shell=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        print("Not implemented yet")


class VrepGoalEnv(gym.GoalEnv):
    def __init__(self, scene, dt):
        self.seed()

        self._dt = dt
        self._port = self.np_random.randint(20000, 21000)
        self._client = run_env(self._port, scene)
        vrep.simxSetFloatingParameter(self._client,
                                      vrep.sim_floatparam_simulation_time_step,
                                      dt, vrep.simx_opmode_blocking)
        print('Connected to port {}'.format(self._port))

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def close(self):
        subprocess.call('pkill -9 vrep &', shell=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        print("Not implemented yet")

    def compute_reward(self, achieved_goal, desired_goal, info):
        return NotImplementedError()
