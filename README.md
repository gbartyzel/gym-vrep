# Open-AI Gym extension for robotics based on V-REP

## Environments

1. Mobile robot navigation - the mobile robot contains ultrasonic sensors,
IMU and two DC motors. Number of ultrasonic sensors depends on task type. The
task of this agent is to navigate from position A to position B. There are
implemented two variants presented below.

|Environment | Description |
| --- | --- |
| RoomNavigation | Environment with static obstacles like walls. |
| DynamicRoomNavigation| Environment with dynamic obstacles like randomly moving mobile robots. |


The environment can be customized with four parameters:
* `navigation_type` - Selects the type of the navigation algorithms from the
supported ones (`ideal`, `odometry`, `gyrodometry`). Default is `ideal`.
* `enable_vision` - Whether to include visual feedback to the observation space.
Default is `False`.
* `goal_threshold` - Defines the threshold value for reaching the goal.
Default value is set to `0.05`.
* `collision_threshold` - Defines the threshold distance for the proximity
sensors, when the collision occurs. Default value is set to `0.05`.

Action space are desired motor angular velocities in rad/s.
They are limited to (0, 15.0) rad/s.

Environment observation space description:
* distances from n proximity sensors
* polar coordinates
* linear and angular velocities
* (optional) image from camera sensor

## Getting started

Here are provided basic requirements for the project:
* CoppeliaSim 4.1.0+
* Python 3.6+
* Ubuntu 20.04
* OpenAI gym
* [PyRep](https://github.com/Souphis/PyRep)

### Python

To install this package run following commands:

```Shell
git clone https://github.com/Souphis/gym-vrep.git
cd gym-vrep
python3 setup.py install
```

### Basic usage

To run the environment with default configuration just add follwing code to your
project:
```Python
import gym
import gym_coppelia_sim

env = gym.make("RoomNavigation-v0") # or env = gym.make("DynamicRoomNavigation-v0")
```

Below is the example how to run environment in the headless mode:

```Python
import gym
import gym_coppelia_sim

env = gym.make("RoomNavigation-v0", headless_mode=True)
```

Changing the navigation algorithm can be done with following example:

```Python
import gym
import gym_coppelia_sim

env = gym.make("RoomNavigation-v0", navigation_algos="odometry")
```
