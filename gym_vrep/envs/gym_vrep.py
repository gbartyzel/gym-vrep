import os
import subprocess
import time
from typing import Any
from typing import NoReturn

import gym
import numpy as np
from gym.utils import seeding

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
        self._scenes_list = os.listdir(
            os.path.join(self._assets_path, 'scenes'))
        self._models_list = os.listdir(
            os.path.join(self._assets_path, 'models'))

        self._dt = dt
        self._port = self.np_random.randint(20000, 21000)
        self._client = self._run_env()

        if not self._get_boolparam(vrep.sim_boolparam_headless):
            self._clear_gui()
        self._load_scene(scene)
        self._load_model(model)

        print('Connected to port {}'.format(self._port))

    def _start(self):
        self._set_floatparam(vrep.sim_floatparam_simulation_time_step, self._dt)

        vrep.simxSynchronous(self._client, True)
        vrep.simxStartSimulation(self._client, vrep.simx_opmode_blocking)

        if not self._get_boolparam(vrep.sim_boolparam_headless):
            self._set_boolparam(vrep.sim_boolparam_threaded_rendering_enabled,
                                True)

    def _run_env(self) -> int:
        """A method that run V-REP in synchronous mode.
        Adding flag '-h' to 'vrep.sh' runs simulation in headless mode.

        Returns: Remote API client ID

        """
        vrep_exec = os.environ['V_REP'] + 'vrep.sh -h '
        synch_mode_cmd = '-gREMOTEAPISERVERSERVICE_' + str(
            self._port) + '_FALSE_TRUE '

        subprocess.call(vrep_exec + synch_mode_cmd + ' &', shell=True)
        time.sleep(5.0)

        print('Connecting to V-REP on port {}'.format(self._port))
        client = vrep.simxStart('127.0.0.1', self._port, True, True, 1000, 5)
        assert client >= 0, 'Connection to V-REP failed!'

        vrep.simxSynchronous(client, True)

        return client

    def _load_scene(self, scene: str) -> NoReturn:
        """Loads given scene in V-REP simulator.

        Args:
            scene: Name of the scene that will be loaded.

        """
        assert (scene + '.ttt' in self._scenes_list), 'Scene not found!'

        scene_path = os.path.join(self._assets_path, 'scenes', scene + '.ttt')
        res = vrep.simxLoadScene(self._client, scene_path, 1,
                                 vrep.simx_opmode_blocking)
        assert (res == vrep.simx_return_ok), 'Could not load scene!'

    def _load_model(self, model) -> NoReturn:
        """Loads given model into V-REP scene

        Args:
            model: Name of the model taht will be loaded.

        """
        assert (model + '.ttm' in self._models_list), 'Model not found!'

        model_path = os.path.join(self._assets_path, 'models', model + '.ttm')
        res, _ = vrep.simxLoadModel(self._client, model_path, 1,
                                    vrep.simx_opmode_blocking)

        assert (res == vrep.simx_return_ok), 'Could not load model!'

    def _clear_gui(self) -> NoReturn:
        """Clears GUI with unnecessary elements like model hierarchy, library
        browser and console. Also this method enables threaded rendering.

        """
        self._set_boolparam(vrep.sim_boolparam_hierarchy_visible, False)
        self._set_boolparam(vrep.sim_boolparam_console_visible, False)
        self._set_boolparam(vrep.sim_boolparam_browser_visible, False)

    def _set_start_pose(self, object_handle: int, pose: np.ndarray) -> NoReturn:
        """Sets start pose of the loaded model.

        Args:
            object_handle: Object handler of the loaded model.
            pose: A model target pose.

        """
        assert (len(np.shape(pose)) == 1), 'Wrong dimension of pose array'
        assert (np.shape(pose)[0] == 6), 'Wrong size of the pose array!'

        res = vrep.simxSetObjectPosition(
            self._client, object_handle, -1, pose[0:3],
            vrep.simx_opmode_blocking)
        assert (res == vrep.simx_return_ok), 'Could not set model position!'

        res = vrep.simxSetObjectOrientation(
            self._client, object_handle, -1, pose[3:],
            vrep.simx_opmode_blocking)
        assert (res == vrep.simx_return_ok), 'Could not set model orientation!'

    def _set_boolparam(self, parameter: int, value: bool) -> NoReturn:
        """Sets boolean parameter of V-REP simulation.

        Args:
            parameter: Parameter to be set.
            value: Boolean value to be set.

        """
        res = vrep.simxSetBooleanParameter(self._client, parameter, value,
                                           vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or
                res == vrep.simx_return_novalue_flag), (
            'Could not set boolean parameter!')

    def _set_floatparam(self, parameter: int, value: float) -> NoReturn:
        """Sets float parameter of V-REP simulation.

        Args:
            parameter: Parameter to be set.
            value: Float value to be set.

        """
        res = vrep.simxSetFloatingParameter(self._client, parameter, value,
                                            vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or
                res == vrep.simx_return_novalue_flag), (
            'Could not set float parameter!')

    def _get_boolparam(self, parameter: int) -> bool:
        res, value = vrep.simxGetBooleanParameter(self._client, parameter,
                                                  vrep.simx_opmode_oneshot)
        assert (res == vrep.simx_return_ok or
                res == vrep.simx_return_novalue_flag), (
            'Could not get boolean parameter!')
        return value

    def step(self, action: np.ndarray) -> Any:
        """A method that perform single simulation step.

        Args:
            action: An action for agent

        """
        raise NotImplementedError

    def reset(self) -> Any:
        """Resets simulation to initial conditions.
        """
        raise NotImplementedError

    def close(self) -> NoReturn:
        """A method that cleanly closes environment.
        """
        vrep.simxStopSimulation(self._client, vrep.simx_opmode_blocking)
        vrep.simxCloseScene(self._client, vrep.simx_opmode_blocking)
        vrep.simxFinish(self._client)
        subprocess.call('pkill -9 vrep &', shell=True)

    def seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'human') -> NoReturn:
        print('Not implemented yet')
