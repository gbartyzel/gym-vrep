## Open-AI Gym extension for robotics based on V-REP

### Environments

1. Mobile robot navigation - the mobile robot contains ultrasonic sensors,
IMU and two DC motors. Number of ultrasonic sensors depends on task type. The
task of this agent is to navigate from position A to position B. Two groups
 of environment variants can be specified:


Static obstacles:

|Environment | Description |
| --- | --- |
| RoomNavigation | Position is obtained from simulation engine. Collision is detected with ultrasonic sensors |
| RoomVisionIdealNavigation | Position is obtained from simulation engine. Collision is detected with camera sensor and ultrasonic sensors|
| RoomOdometryNavigation | Position is obtained from encoders ticks. Collision is detected with ultrasonic sensors |
| RoomGyrodometryNavigation | Position is obtained from encoders ticks and gyroscope. Collision is detected with ultrasonic sensors |

Dynamic obstacles:

|Environment | Description |
| --- | --- |
| DynamicRoomNavigation| Position is obtained from simulation engine. Collision is detected with ultrasonic sensors. Moving robots in environment as dynamic obstacles. |
| DynamicVisionRoomNavigation | Position is obtained from simulation engine. Collision is detected with camera sensor and ultrasonic sensors. Moving robots in environment as dynamic obstacles. |

Action space are desired motor angular velocities in rad/s. They are limited
 to (0, 15.0)[rad/s].

Environment state space description:

| Classic envs:                      | Vision envs:                       |
| ---------------------------------- | ---------------------------------- |
| distances from n proximity sensors | distances from n proximity sensors |
| polar coordinates                  | polar coordinates                  |
| linear and angular velocities      | image from camera sensor           |
|                                    | linear and angular velocities      |

### Installation

#### Requirements:
Basic requirements:
* CoppeliaSim 4.1.0+
* Python 3.6+
* Ubuntu 20.04
* OpenAI gym
* [PyRep](https://github.com/Souphis/PyRep)

#### Python
```
git clone https://github.com/Souphis/gym-vrep.git
cd gym-vrep
python3 setup.py install
```
