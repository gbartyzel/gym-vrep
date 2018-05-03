from setuptools import setup
import sys, os.path

# Don't import gym module here, since deps may not be installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym-vrep'))

setup(name='gym-vrep',
      version='0.0.1',
      install_requires=['gym>=0.2.3'],
      description='The OpenAI Gym for robotics.',
      url='https://github.com/souphis/gym-vrep',
      author='Grzegorz Bartyzel',
      package_data={'gym-vrep': ['envs/scenes/*.ttt']},
)
