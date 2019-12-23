import os
from typing import NoReturn

import gym
from gym.utils import seeding
from pyrep import PyRep
from pyrep.backend import sim, simConst


class VrepEnv(gym.Env):
    """
    Superclass for V-REP gym environments.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 scene: str,
                 dt: float,
                 model: str = None,
                 headless_mode: bool = False):
        """
        Class constructor
        Args:
            scene: String, name of the scene to be loaded
            dt: Float, a time step of the simulation
            model:  Optional[String], name of the model that to be imported
            headless_mode: Bool, define mode of the simulation
        """
        self.seed()
        self._assets_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'assets')
        self._scenes_path = os.path.join(self._assets_path, 'scenes')
        self._models_path = os.path.join(self._assets_path, 'models')

        self._pr = PyRep()
        self._launch_scene(scene, headless_mode)
        self._import_model(model)
        if not headless_mode:
            self._clear_gui()
        self._pr.set_simulation_timestep(dt)
        self._pr.start()

    def _launch_scene(self, scene: str, headless_mode: bool):
        assert os.path.splitext(scene)[1] == '.ttt'
        assert scene in os.listdir(self._scenes_path)
        scene_path = os.path.join(self._scenes_path, scene)
        self._pr.launch(scene_path, headless=headless_mode)

    def _import_model(self, model: str):
        if model is not None:
            assert os.path.splitext(model)[1] == '.ttm'
            assert model in os.listdir(self._models_path)
            model_path = os.path.join(self._models_path, model)
            self._pr.import_model(model_path)

    @staticmethod
    def _clear_gui():
        sim.simSetBoolParameter(simConst.sim_boolparam_browser_visible, False)
        sim.simSetBoolParameter(simConst.sim_boolparam_hierarchy_visible, False)
        sim.simSetBoolParameter(simConst.sim_boolparam_console_visible, False)

    def step(self, action):
        return NotImplementedError

    def reset(self):
        return NotImplementedError

    def close(self):
        self._pr.stop()
        self._pr.shutdown()

    def seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode: str = 'human') -> NoReturn:
        print('Not implemented yet')
