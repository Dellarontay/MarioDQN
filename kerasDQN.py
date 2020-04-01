# pip install gym-super-mario-bros
from PIL import Image
import cv2
from matplotlib import pyplot as plt


import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym_super_mario_bros.actions import RIGHT_ONLY


from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

import sys
import tensorflow as tf
import networkx as nx
import math
import random
import copy
import numpy as np
import pylab as plt

import pandas

env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

def cnn_model_fn(features,labels,mode):
    # NOTE Model function for CNN.
    # Input Layer
    input_layer = tf.reshape(features["X"],[-1,240,256,1])

    # Convulational layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32
    )


def main():
    pass

if __name__ == '__main__':
    main()