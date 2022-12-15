import os
from typing import Optional

import gym
import numpy as np
from pyrep import PyRep
from pyrep.backend import sim, simConst

from gym_coppelia_sim.common.typing import ArrayStruct, EnvironmentTuple


class CoppeliaSimEnv(gym.Env):
    """
    Superclass for CoppeliaSim gym environments.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        scene: str,
        dt: float,
        model: Optional[str] = None,
        headless_mode: bool = False,
    ):
        """Initialize class object.

        Args:
            scene: String, name of the scene to be loaded
            dt: Float, a time step of the simulation
            model:  Optional[String], name of the model that to be imported
            headless_mode: Bool, define mode of the simulation
        """
        self._assets_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "assets"
        )
        self._scenes_path = os.path.join(self._assets_path, "scenes")
        self._models_path = os.path.join(self._assets_path, "models")

        self._pr = PyRep()
        self._launch_scene(scene, headless_mode)
        self._import_model(model)
        if not headless_mode:
            self._clear_gui()
        self._pr.set_simulation_timestep(dt)
        self._pr.start()

    def _launch_scene(self, scene: str, headless_mode: bool):
        assert os.path.splitext(scene)[1] == ".ttt"
        assert scene in os.listdir(self._scenes_path)
        scene_path = os.path.join(self._scenes_path, scene)
        self._pr.launch(scene_path, headless=headless_mode)

    def _import_model(self, model: Optional[str]):
        if model is not None:
            assert os.path.splitext(model)[1] == ".ttm"
            assert model in os.listdir(self._models_path)
            model_path = os.path.join(self._models_path, model)
            self._pr.import_model(model_path)

    def _clear_gui(self):
        sim.simSetBoolParameter(simConst.sim_boolparam_browser_visible, False)
        sim.simSetBoolParameter(simConst.sim_boolparam_hierarchy_visible, False)
        sim.simSetBoolParameter(simConst.sim_boolparam_console_visible, False)

    def step(self, action: np.ndarray) -> EnvironmentTuple:
        raise NotImplementedError

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> ArrayStruct:
        super().reset(seed=seed, return_info=return_info, options=options)

    def close(self):
        self._pr.stop()
        self._pr.shutdown()

    def render(self, mode: str = "human"):
        print("Not implemented yet")
