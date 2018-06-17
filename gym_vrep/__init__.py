import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(
    id='VrepMobileRobotIdealNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:VrepMobileRobotNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='VrepMobileRobotOdometryNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:VrepMobileRobotOdomNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='VrepMobileRobotGyrodometryNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:VrepMobileRobotGyroNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='VrepMobileRobotNavigationGoal-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:VrepMobileRobotNavigationGoalEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)
