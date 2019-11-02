import gym
import gym_vrep
import multiprocessing as mp
import numpy as np


def run(i):
    np.random.seed(i)
    env = gym.make('MobileRobotVisionIdealNavigation-v0')

    for _ in range(2):
        env.reset()
        for j in range(1000):
            state, reward, done, _ = env.step([5.0, 5.0])
            # print('Agent: {}, state: {}'.format(i, state))
            if done:
                break
    env.close()


processes = [mp.Process(target=run, args=(i,)) for i in range(1)]

for p in processes:
    p.start()

for p in processes:
    p.join()
