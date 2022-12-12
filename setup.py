import os.path
import sys

from setuptools import find_packages, setup

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym-coppelia-sim"))

__version__ = "0.2.0"

setup(
    name="gym_coppelia_sim",
    version=__version__,
    license="Apache License 2.0",
    install_requires=[
        "gym==0.22.0",
        "numpy",
        "opencv-python==4.2.*",
    ],
    description="The OpenAI Gym for robotics. Toolkit is using CoppeliaSim.",
    url="https://github.com/souphis/gym_vrep",
    author="Grzegorz Bartyzel",
    author_email="gbartyzel@hotmail.com",
    packages=find_packages(exclude=("test",)),
    include_package_data=True,
    package_data={
        "gym_coppelia_sim": ["envs/assets/scenes/*.ttt", "envs/assets/models/*ttm"]
    },
)
