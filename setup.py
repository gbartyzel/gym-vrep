from setuptools import setup, find_packages
import sys
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym-vrep'))

setup(
      name='gym_vrep',
      version='1.1.0',
      license='Apache License 2.0',
      install_requires=['gym>=0.10.5', 'numpy>=1.13'],
      description='The OpenAI Gym for robotics. Toolkit is using V-REP.',
      url='https://github.com/souphis/gym_vrep',
      author='Grzegorz Bartyzel',
      author_email='gbartyzel@hotmail.com',
      packages=find_packages(exclude=('test',)),
      include_package_data=True,
      package_data={'gym-vrep': ['envs/assets/scenes/*.ttt', 'envs/assets/models/*ttm']}
)
