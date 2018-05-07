from setuptools import setup
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'gym-vrep'))

setup(name='gym-vrep',
      version='0.0.1',
      install_requires=['gym>=0.10.0', 'numpy>=1.13'],
      description='The OpenAI Gym for robotics. Toolkit is using V-REP.',
      url='https://github.com/souphis/gym-vrep',
      author='Grzegorz Bartyzel',
      package_data={'gym-vrep': ['envs/scenes/*.ttt']},
)
