## Open-AI Gym extension for robotics based on V-REP

### Environments

1. Mobile robot navigation - the mobile robot contains five proximity sensors,
IMU and two DC motors. The task of this agent is to navigate from position A
 to position B. Four variants of this problem were implemented:

| Environment                      | Description                                                                                         |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| MobileRobotIdealNavigation       | Position is obtained from simulation engine. Collision is detected with proximity sensor            |
| MobileRobotVisualIdealNavigation | Position is obtained from simulation engine. Collision is detected with camera sensor               |
| MobileRobotOdometryNavigation    | Position is obtained from encoders ticks. Collision is detected with proximity sensor               |
| MobileRobotGyrodometryNavigation | Position is obtained from encoders ticks and gyroscope. Collision is detected with proximity sensor |

Action space are desired motor angular velocities in rad/s. They are limited to (0, 10.0)[rad/s].

Environment state space description:

| Classic env:                       | Visual env:                        |
| ---------------------------------- | ---------------------------------- |
| distances from 5 proximity sensors | distances from 5 proximity sensors |
| polar coordinates                  | polar coordinates                  |
| linear and angular velocities      | image from camera sensor           |
|                                    | linear and angular velocities      |

Environment reward:

![equation](https://latex.codecogs.com/gif.latex?R%20%3D%20V_L%5Ccdot%20%5Ccos%5Ctheta%5Ccdot%20%5Cmin%28D%29)

Where

![equation](https://latex.codecogs.com/gif.latex?V_L) - linear velocity of the mobile robot

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta)- heading angle of the mobile robot

![equation](https://latex.codecogs.com/gif.latex?D) - vector od distances read from proximity 
sensors

    
### Installation

#### Requirements:
Basic requirements:
* V-REP 3.5.0
* Python 3.6+
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
