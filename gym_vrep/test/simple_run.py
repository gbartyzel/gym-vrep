import gym
import gym_vrep
import multiprocessing as mp
import numpy as np


def run(i):
    np.random.seed(i)
    env = gym.make('MobileRobotIdealNavigation-v0')
    print(env.observation_space.size)
    for _ in range(1):
        env.reset()
        for j in range(1000):
            state, reward, done, _ = env.step(env.action_space.sample())
            if done:
                break
    env.close()


processes = [mp.Process(target=run, args=(i,)) for i in range(1)]

for p in processes:
    p.start()

for p in processes:
    p.join()
