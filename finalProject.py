# Legitimate Project ideas
# Using OpenAI gym to as a simulator for Mario RL algorithm camprison with small tweaks.
# Might also want to work on a project dealing with turbulenc
# pip install gym-super-mario-bros
from PIL import Image
import cv2
from matplotlib import pyplot as plt


import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY


from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv, wrap as nes_wrap

import sys

import networkx as nx
import math
import random
import copy
import numpy as np
import pylab as plt

import pandas

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)


def printObject(D, filename):
    with open(filename, 'w') as f:
        for x in range(0, len(D)):
            # print(len(D))
            # print(len(D[0]))
            # input("wait")
            for y in range(0, len(D[x])):
                if y == len(D[x]) - 1:
                    f.write("{}\n".format(D[x][y]))
                else:
                    f.write("{}".format(D[x][y]))

def test():
    # stateMap = {x:np.zeros(6) for x in range(65536)}
    pass

def findMario(state):  
    img_gray = cv2.cvtColor(state,6) # 6 is gray
    print(img_gray)
    printObject(img_gray,"gray_img.txt")
    input("wait")
    template = cv2.imread('sprites/mario/mario-small-right.png',0)
    # template = cv2.imread('sprites/misc/overworld/rock.png',0) #Floor

    # cv2.imwrite('color_img.png', img_gray)
    # cv2.imshow("image", img_gray);
    # input("wait")
    print(template.shape[::-1])
    print(img_gray.shape[::-1])
    # w, h = template.shape[::-1]
    w=16
    h=16

    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res>=threshold)
    newState = state.copy()
    print(loc)
    Mario = (0,1)
    for pt in zip(*loc[::-1]):
        # pass
        print("point: ",pt)
        Mario = pt
        cv2.rectangle(newState,pt,(pt[0] + h, pt[1] +w), (0,0,255),2)
    cv2.imwrite('res.png',newState)
    return Mario

def main():
    print(RIGHT_ONLY)
    print(SIMPLE_MOVEMENT)
    print(COMPLEX_MOVEMENT)
    # Gym-super-mario Code
    learningRate = 0.01
    discount = 0.95
    actionSpace = 12 + 1
    # State space can be abstracted to 5x5 grid
    stateSpace = 3161+1
    Q = np.zeros((stateSpace, actionSpace))
    done = True
    # 3161 x position till flag
    for episode in range(5):
        for step in range(5000):
            if done:
                state = env.reset()
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            # print(env.observation_space.shape[0])
            # Mario = findMario(state)
            # input("wait MarAI")
            # info.x_pos

            if info["x_pos"] == 3161:
                max_a_Q = max(Q[info["x_pos"], :])
            else:
                max_a_Q = max(Q[info["x_pos"] + 1, :])

            Q[info["x_pos"]][action] += learningRate * \
                (reward + discount * max_a_Q - Q[info["x_pos"]][action])
            if step % 50 == 0:
                print(info)
                pos = info["x_pos"]
                print(pos)
                pos = 16*2
                newImg = state[ 0:239][191:255]
                # newImg = state[191:200][:]
                # multiple = 16
                # for it in range(0,len(state)):
                #     newImgIter = 0
                #     for y in range(info["x_pos"],info["x_pos"] + 16):
                #         newImg[it][newImgIter] = state[it][y]
                #         newImgIter+=1
                #         if newImgIter == multiple :
                #             newImgiter = 0
                print(info)
                printObject(state, "object.txt")
                img = Image.fromarray(newImg, 'RGB')
                # img.save('my.png')
                img.show()
                input("wait")

            env.render()
        print(episode)
        # gym super mario code
    lastPos = -1
    for step in range(5000):
        if done:
            state = env.reset()
        # state, reward, done, info = env.step(env.action_space.sample())
        # if step != 5000:
        #     max_a_Q = max(Q[step+1,:])
        # else:
        #     max_a_Q = max(Q[step,:])
        if step == 0:
            actions = Q[40][:]
        else:
            actions = Q[lastPos][:]
        action = np.argmax(actions)
        state, reward, done, info = env.step(action)
        lastPos = info["x_pos"]

        # if lastStep == -1:
        #     lastStep = info["x_pos"]
        # else:

        env.render()

        # Q[s][a]
            #     max_a_Q = max(Q[row.sp, :])
            # currState = row.s
            # currAction = row.a

            # Q[currState][currAction] += learningRate * \
            #     (discount * (row.r + discount *
            #                  max_a_Q - Q[currState][currAction]))

    env.close()

    # if len(sys.argv) != 2:
    #     raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    # inputfilename = sys.argv[1]
    # Not as simple as running over one file
    # beginSearch(inputfilename)


if __name__ == '__main__':
    main()
