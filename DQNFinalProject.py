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


from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

import sys

import networkx as nx
import math
import random
import copy
import numpy as np
import pylab as plt

import pandas
import tensorflow as tf

from keras.models import Sequential
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)


def main():
    with tf.device("/gpu:0"):
        mnist = tf.keras.datasets.mnist

        (x_train, y_train),(x_test, y_test) = mnist.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5)
        model.evaluate(x_test, y_test)


if __name__ == '__main__':
    main()
