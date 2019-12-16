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

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20R%20%3D%20%5Cleft%5Cbegin%7BBmatrix%7D%201%20%26%20%2Cd%20%3C%20d_%7Bth%7D%5C%5C%20-1%20%26%20%2C%5Cexists%7BD%7D%3Cd_%7Bcollision%7D%5C%5C%20-0.1%20%26%20%2C%5Cexists%7BD%7D%20%3C%20d_%7Bproxth%7D%5C%5C%20V_L%5Ccdot%5Ccos%7B%5Ctheta%7D%20%26%20%2Cotherwise%20%5Cend%7Bmatrix%7D%5Cright.)

Where

![equation](https://latex.codecogs.com/gif.latex?V_L) - linear velocity of the mobile robot

![equation](https://latex.codecogs.com/gif.latex?%5Ctheta)- heading angle of the mobile robot

![equation](https://latex.codecogs.com/gif.latex?D) - vector od distances
 read from ultrasonic sensors
 
![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20d) - distance
 between robot and target position
 
![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20d_%7Bth%7D
) - threshold distance between robot and target position

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20d_%7Bcollision%7D
) - distance when collision occurs

![equation](https://latex.codecogs.com/gif.latex?%5Clarge%20d_%7Bproxth%7D
) - safety distance threshold

    
### Installation

#### Requirements:
Basic requirements:
* V-REP 3.5.0
* Python 3.6+
* Ubuntu 16.04 / Arch Linux
* OpenAI gym
* [PyRep](https://github.com/Souphis/PyRep)


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
