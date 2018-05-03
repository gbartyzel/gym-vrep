import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(
    id='VrepMobileRobotIdealNavigation-v0',
    entry_point='gym_vrep.envs.mobile_robot_navigation:VrepMovileRobotIdealNavigationEnv',
    max_episode_steps=2400,
)
