# pip install gym-pull
# import gym_super_mario_bros
# from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
# from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
# from gym_super_mario_bros.actions import RIGHT_ONLY
# from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym
import gym_pull
import gym.configuration
import sys

import networkx as nx
import math
import random
import copy
import numpy as np
import pylab as plt

import pandas


gym_pull.pull('github.com/ppaquette/gym-super-mario')        # Only required once, envs will be loaded with import gym_pull afterwards
env = gym.make('ppaquette/SuperMarioBros-1-1-v0')

# env = gym_super_mario_bros.make('SuperMarioBros-v1')
# env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

def printObject(D, filename):
    with open(filename, 'w') as f:
        for x in range(0,len(D)):
            lineCounter = 1
            for y in range(0,len(D[x])):
                if lineCounter == 3:
                    f.write("{}\n".format(D[x][y]))
                    lineCounter = 0
                else:
                    f.write("{}".format(D[x][y]))
                lineCounter +=1


def main():
    learningRate = 0.01
    discount = 0.95
    actionSpace = 12 +1
    # State space can be abstracted to 5x5 grid
    stateSpace = 5000+1
    Q = np.zeros((5000,actionSpace))
    done = True
    for episode in range (5):
        for step in range(5000):
            if done:
                state = env.reset()
            action = env.action_space.sample()
            print(action)
            input("wait")
            state, reward, done, info = env.step(action)
            if step == 4999:
                max_a_Q = max(Q[step,:])
            else:
                max_a_Q = max(Q[step+1,:])

            Q[step][action] += learningRate * (reward + discount * max_a_Q - Q[step][action])
            if step% 50 ==0:
                printObject(state,"object.txt")
                input("wait")
            # env.render()
        print(episode)
        # gym super mario code
    # for step in range(5000):
    #     if done:
    #         state = env.reset()
    #     actions = Q[step][:]
    #     action = np.argmax(actions)
    #     state, reward, done, info = env.step(action)
    #     env.render()
    env.close()

if __name__ == '__main__':
    main()
