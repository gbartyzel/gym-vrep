import gym
import gym_vrep

env = gym.make('MobileRobotIdealNavigation-v0')

for _ in range(5):
    env.reset()
    for i in range(1000):
        env.step([0.0, 0.0])

env.close()
