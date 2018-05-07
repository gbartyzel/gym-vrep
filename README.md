# Open-AI Gym extension for robotics based on V-REP

## Environments

1. Mobile robot navigation:
* description: Mobile robot is a learning agent. It contains five proximity 
sensors, IMU and two DC motors. The task of this agent is to navigate from 
position A to position B.
* variants:
    * ideal - position is obtained from simulation engine
    * odometry - position is obtained from encoders ticks
    * gyrodometry - position is obtained from gyroscope and encoders ticks
* state:
    * classic env:
        * reading from 5 proximity sensors
        * polar coordinates
    * goal based env:
        * observation:
            * reading from 5 proximity sensors
            * robot cartesian position
        * achieved goal:
            * robot cartesian position
        * desired goal:
            * goal cartesian position
    
## Installation

### Requirements:
Basic requirements:
* V-REP 3.5.0
* Python 3.5+
* Ubuntu 16.04 / Arch Linux
* OpenAI gym


### V-REP
```
cd ~
wget http://coppeliarobotics.com/files/V-REP_PRO_EDU_V3_5_0_Linux.tar.gz
tar -xzvf V-REP_PRO_EDU_V3_5_0_Linux 
sudo mv V-REP_PRO_EDU_V3_5_0_Linux /opt/V-REP
rm -rf V-REP_PRO_EDU_V3_5_0_Linux
export V_REP=/opt/V-REP
```
### Python
```
git clone https://github.com/Souphis/gym-vrep.git
cd gym-vrep
pip install -e .
```
