import os
import subprocess
import time

import gym
from gym.utils import seeding
from typing import NoReturn, Dict, Any
from numpy import ndarray

from gym_vrep.envs.vrep import vrep


class VrepEnv(gym.Env):
    """
    Superclass for V-REP gym environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, scene: str, model: str, dt: float):
        """Initialize simulation in V-REP.

        Args:
            scene: A scene that will be load in V-REP
            model: A agent model that will be load in V-REP
            dt: A delta time of simulation
        """
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

    def _run_env(self) -> int:
        """A method that run V-REP in synchronous mode.
        Adding flag '-h' to 'vrep.sh' runs simulation in headless mode.

        Returns:

        """
        vrep_exec = os.environ['V_REP'] + 'vrep.sh '
        synch_mode_cmd = '-gREMOTEAPISERVERSERVICE_' + str(self._port) + '_FALSE_TRUE '

        subprocess.call(vrep_exec + synch_mode_cmd + ' &', shell=True)
        time.sleep(5.0)

        client = vrep.simxStart('127.0.0.1', self._port, True, True, 1000, 5)
        vrep.simxSynchronous(client, True)

        return client

    def _load_environment(self, scene: str) -> NoReturn:
        scene_path = os.path.join(self._assets_path, 'scenes', scene + '.ttt')
        vrep.simxLoadModel(self._client, scene_path, 1, vrep.simx_opmode_blocking)

    def _clear_gui(self) -> NoReturn:
        self._set_boolparam(vrep.sim_boolparam_hierarchy_visible, False)
        self._set_boolparam(vrep.sim_boolparam_console_visible, False)
        self._set_boolparam(vrep.sim_boolparam_browser_visible, False)
        self._set_boolparam(vrep.sim_boolparam_threaded_rendering_enabled, True)

    def _spawn_robot(self) -> NoReturn:
        model_path = os.path.join(self._assets_path, 'models', self._model + '.ttm')
        vrep.simxLoadModel(self._client, model_path, 1, vrep.simx_opmode_blocking)

    def _set_start_pose(self, object_handle: int, pose: Dict[str, ndarray]) -> NoReturn:
        vrep.simxSetObjectPosition(
            self._client, object_handle, -1, pose['position'], vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(
            self._client, object_handle, -1, pose['orientation'], vrep.simx_opmode_blocking)

    def _set_boolparam(self, parameter: int, value: bool) -> NoReturn:
        vrep.simxSetBooleanParameter(self._client, parameter, value, vrep.simx_opmode_oneshot)

    def _set_floatparam(self, parameter: int, value: float) -> NoReturn:
        vrep.simxSetFloatingParameter(self._client, parameter, value, vrep.simx_opmode_oneshot)

    def _get_boolparam(self, parameter: int) -> bool:
        res, value = vrep.simxGetBooleanParameter(self._client, parameter, vrep.simx_opmode_oneshot)
        return value

    def step(self, action: ndarray) -> Any:
        raise NotImplementedError

    def reset(self) -> Any:
        raise NotImplementedError

    def close(self) -> NoReturn:
        vrep.simxStopSimulation(self._client, vrep.simx_opmode_blocking)
        vrep.simxFinish(self._client)
        subprocess.call('pkill -9 vrep &', shell=True)

    def seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'human') -> NoReturn:
        print("Not implemented yet")
