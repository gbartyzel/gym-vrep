import numpy as np

from gym_coppelia_sim.common.typing import NumpyOrFloat


def correct_angle(angle: NumpyOrFloat) -> NumpyOrFloat:
    """Functin that correct given angles into range -pi, pi

    Args:
        angle: An array of angles or single angle

    Returns:
        new_angle: Angles or angle after correction.
    """
    new_angle = None
    if angle >= 0:
        new_angle = np.fmod((angle + np.pi), (2 * np.pi)) - np.pi

    if angle < 0:
        new_angle = np.fmod((angle - np.pi), (2 * np.pi)) + np.pi

    new_angle = np.round(angle, 3)
    return new_angle
