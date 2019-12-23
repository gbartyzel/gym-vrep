import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(
    id='RoomNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:NavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='RoomVisionNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:VisionNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='DynamicRoomNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:DynamicNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='DynamicRoomVisionNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:DynamicVisionNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='RoomOdometryNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:NavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)

register(
    id='RoomGyrodometryNavigation-v0',
    entry_point=
    'gym_vrep.envs.mobile_robot_navigation:GyroNavigationEnv',
    max_episode_steps=1200,
    kwargs={'dt': 0.05},
)
