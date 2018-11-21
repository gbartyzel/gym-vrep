import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(
    id='MobileRobotIdealNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:MobileRobotNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='MobileRobotVisionIdealNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:MobileRobotVisionNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='MobileRobotOdometryNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:MobileRobotOdomNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='MobileRobotGyrodometryNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:MobileRobotGyroNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)
