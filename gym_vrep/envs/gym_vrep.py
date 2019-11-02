import os
import subprocess
import time
from typing import Any
from typing import NoReturn
from typing import Tuple

import gym
import numpy as np
import psutil
from gym.utils import seeding

import gym_vrep.envs.vrep.vrep_lib as vlib


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
        self._scenes_path = os.path.join(self._assets_path, 'scenes')
        self._models_path = os.path.join(self._assets_path, 'models')

        self._dt = dt
        self._port = self.np_random.randint(20000, 21000)
        self._client, self._process = self._run_env()

        if not vlib.is_headless(self._client):
            self._clear_gui()
        vlib.load_scene(self._client, scene, self._scenes_path)
        vlib.load_model(self._client, model, self._models_path)

        print('Connected to port {}'.format(self._port))

    def _start(self):
        vlib.set_simulation_time_step(self._client, self._dt)
        vlib.start_simulation(self._client, True)
        if not vlib.is_headless(self._client):
            vlib.set_thread_rendering(self._client, True)

    def _run_env(self) -> Tuple[int, subprocess.Popen]:
        """A method that run V-REP in synchronous mode.
        Adding flag '-h' to 'vrep.sh' runs simulation in headless mode.

        Returns: Remote API client ID

        """
        cmd = [
            os.environ['V_REP'] + 'vrep.sh', '-h',
            '-gREMOTEAPISERVERSERVICE_' + str(self._port) + '_FALSE_TRUE', '&'
        ]

        process = subprocess.Popen(cmd)
        time.sleep(5.0)

        client = vlib.connect(self._port, True)

        return client, process

    def _clear_gui(self) -> NoReturn:
        """Clears GUI with unnecessary elements like model hierarchy, library
        browser and console. Also this method enables threaded rendering.

        """
        vlib.set_hierarchy_visibility(self._client, False)
        vlib.set_browser_visbility(self._client, False)
        vlib.set_console_visibility(self._client, False)

    def _set_start_pose(self, object_handle: int, pose: np.ndarray) -> NoReturn:
        """Sets start pose of the loaded model.

        Args:
            object_handle: Object handler of the loaded model.
            pose: A model target pose.

        """
        assert (len(np.shape(pose)) == 1), 'Wrong dimension of pose array'
        assert (np.shape(pose)[0] == 6), 'Wrong size of the pose array!'
        vlib.set_object_position(self._client, object_handle, pose[:3])
        vlib.set_object_orientation(self._client, object_handle, pose[3:])

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
        vlib.disconnect(self._client)
        parent = psutil.Process(self._process.pid)
        for p in parent.children(recursive=True):
            p.kill()
        self._process.kill()

    def seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'human') -> NoReturn:
        print('Not implemented yet')
