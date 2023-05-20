"""
import unittest

from gym_coppelia_sim.envs.mobile_robot_navigation import NavigationEnv
from gym_coppelia_sim.envs.mobile_robot_navigation.navigation_algos import (
    GyrodometryNavigationAlgorithm,
    IdealNavigationAlgorithm,
    NavigationAlgorithm,
    OdometryNavigationAlgorithm,
)


class TestNavigationAlgos(unittest.TestCase):
    def test_build_factory(self):
        env = NavigationEnv(headless_mode=True)

        algo = NavigationAlgorithm.build(
            "ideal",
            dt=0.05,
        )
        self.assertIsInstance(algo, IdealNavigationAlgorithm)

        algo = NavigationAlgorithm.build("odometry")
        self.assertIsInstance(algo, OdometryNavigationAlgorithm)

        algo = NavigationAlgorithm.build("gyrodometry")
        self.assertIsInstance(algo, GyrodometryNavigationAlgorithm)
"""


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main(["-v", __file__]))
