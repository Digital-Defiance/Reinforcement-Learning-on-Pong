import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

# icon = cv2.imread("paddle.png") / 255.0
icon = cv2.imread("paddle.png") 
icon = cv2.resize(icon, (800, 400))

print(icon.shape)