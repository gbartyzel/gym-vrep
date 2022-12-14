import logging

from gym.envs.registration import register

logger = logging.getLogger(__name__)
register(
    id="RoomNavigation-v0",
    entry_point="gym_coppelia_sim.envs.mobile_robot_navigation:NavigationEnv",
    max_episode_steps=1200,
    kwargs={"dt": 0.05},
)

register(
    id="DynamicRoomNavigation-v0",
    entry_point="gym_coppelia_sim.envs.mobile_robot_navigation:DynamicNavigationEnv",
    max_episode_steps=1200,
    kwargs={"dt": 0.05},
)
