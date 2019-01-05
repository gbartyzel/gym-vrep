import os
import subprocess
import time

import gym
from gym.utils import seeding

from gym_vrep.envs.vrep import vrep


class VrepEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, scene, model, dt):
        self.seed()
        self._assets_path = os.path.join(os.path.dirname(__file__), 'assets')
        self._model = model

        self._dt = dt
        self._port = self.np_random.randint(20000, 21000)
        self._client = self._run_env()

        self._set_floatparam(vrep.sim_floatparam_simulation_time_step, dt)

        self._load_environment(scene)
        self._spawn_robot()
        if not self._get_boolparam(vrep.sim_boolparam_headless):
            self._clear_gui()

        print('Connected to port {}'.format(self._port))

    def _run_env(self):
        vrep_exec = os.environ['V_REP'] + 'vrep.sh '
        synch_mode_cmd = '-gREMOTEAPISERVERSERVICE_' + str(self._port) + '_FALSE_TRUE '

        subprocess.call(vrep_exec + synch_mode_cmd + ' &', shell=True)
        time.sleep(5.0)

        client = vrep.simxStart('127.0.0.1', self._port, True, True, 1000, 5)
        vrep.simxSynchronous(client, True)

        return client

    def _load_environment(self, scene):
        scene_path = os.path.join(self._assets_path, 'scenes', scene + '.ttt')
        vrep.simxLoadModel(self._client, scene_path, 1, vrep.simx_opmode_blocking)

    def _clear_gui(self):
        self._set_boolparam(vrep.sim_boolparam_hierarchy_visible, False)
        self._set_boolparam(vrep.sim_boolparam_console_visible, False)
        self._set_boolparam(vrep.sim_boolparam_browser_visible, False)
        self._set_boolparam(vrep.sim_boolparam_threaded_rendering_enabled, True)

    def _spawn_robot(self):
        model_path = os.path.join(self._assets_path, 'models', self._model + '.ttm')
        vrep.simxLoadModel(self._client, model_path, 1, vrep.simx_opmode_blocking)

    def _set_start_pose(self, object_handle, pose):
        vrep.simxSetObjectPosition(
            self._client, object_handle, -1, pose['position'], vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(
            self._client, object_handle, -1, pose['orientation'], vrep.simx_opmode_blocking)

    def _set_boolparam(self, parameter, value):
        vrep.simxSetBooleanParameter(self._client, parameter, value, vrep.simx_opmode_oneshot)

    def _set_floatparam(self, parameter, value):
        vrep.simxSetFloatingParameter(self._client, parameter, value, vrep.simx_opmode_oneshot)

    def _get_boolparam(self, parameter):
        res, value = vrep.simxGetBooleanParameter(self._client, parameter, vrep.simx_opmode_oneshot)
        return value

    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def close(self):
        vrep.simxStopSimulation(self._client, vrep.simx_opmode_blocking)
        vrep.simxFinish(self._client)
        subprocess.call('pkill -9 vrep &', shell=True)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        print("Not implemented yet")
