## Open-AI Gym extension for robotics based on V-REP

### Environments

1. Mobile robot navigation - the mobile robot contains five proximity sensors,
IMU and two DC motors. The task of this agent is to navigate from position A to position B.
Four variants of this problem were implemented:

| Environment                      | Description                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| MobileRobotIdealNavigation       | Position is obtained from simulation engine. Collision is detected with proximity sensor            |
| MobileRobotVisualIdealNavigation | Position is obtained from simulation engine. Collision is detected with camera sensor               |
| MobileRobotOdometryNavigation    | Position is obtained from encoders ticks. Collision is detected with proximity sensor               |
| MobileRobotGyrodometryNavigation | Position is obtained from encoders ticks and gyroscope. Collision is detected with proximity sensor |

Environment state description:

| Classic env:                       | Visual env:                        |
| ---------------------------------- | ---------------------------------- |
| distances from 5 proximity sensors | distances from 5 proximity sensors |
| polar coordinates                  | polar coordinates                  |
| linear and angular velocities      | image from camera sensor           |
|                                    | linear and angular velocities      |


    
### Installation

#### Requirements:
Basic requirements:
* V-REP 3.5.0
* Python 3.5+
* Ubuntu 16.04 / Arch Linux
* OpenAI gym


#### V-REP
```
chmod +x ./install_vrep.sh
./install_vrep.sh
```
#### Python
```
git clone https://github.com/Souphis/gym-vrep.git
cd gym-vrep
pip install -e .
```
