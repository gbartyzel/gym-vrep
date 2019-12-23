import gym
import gym_vrep
import cv2


def main():
    env = gym.make('DynamicRoomNavigation-v0')

    for i in range(10):
        env.reset()
        for i in range(1000):
            state, reward, done, _ = env.step(env.action_space.sample())
            if done:
                break

    env.close()


if __name__ == '__main__':
    main()
