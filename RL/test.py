import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

import warnings
warnings.filterwarnings('ignore')


font = cv2.FONT_HERSHEY_COMPLEX_SMALL



class Striker:
    def __init__(self):
        self.x = 0
        self.y = 200
        self.width = 15
        self.height = 60
        self.y_min = 0
        self.y_max = 400
        self.velocity = 5
        self.screen_height = 400

        # now creating icon for it
        self.icon = cv2.imread("paddle.png") / 255.0
        self.icon = cv2.resize(self.icon, (self.width, self.height))
        
        

    # def set_position(self, x, y):
    #     self.x = x
    #     self.y = y

    def get_position(self):
        return (self.x, self.y)
    
    def move(self,direction):
        if direction == 'Up':
            self.y -= self.velocity
        elif direction == 'Down':
            self.y += self.velocity
        
        if self.y < 0:
            self.y = 0
        elif self.y + self.height > self.screen_height:
            self.y = self.screen_height - self.height

        # self.set_position(self.x, self.y)
             
        

class Ball:
    def __init__(self):
        self.x = 400
        self.y = 200
        self.radius = 10
        self.velocity_x = 3
        self.velocity_y = 3
        self.color = (255, 255, 255)
        self.screen_width = 800
        self.screen_height = 400

        self.icon = cv2.imread("ball.png") / 255.0
        self.icon = cv2.resize(self.icon, (self.radius, self.radius))


    # def set_position(self, x, y):
    #     self.x = x
    #     self.y = y

    def get_position(self):
        return (self.x, self.y)


    def move(self):
        self.x -= self.velocity_x
        self.y -= self.velocity_y

        # if self.y - self.radius <= 0 or self.y + self.radius >= self.screen_height:
        #     self.velocity_y = -self.velocity_y
        # if self.x - self.radius <= 0 or self.x + self.radius >= self.screen_width:
        #     self.velocity_x =  -self.velocity_x

        if self.y + self.radius >= self.screen_height:
            self.velocity_y = -self.velocity_y
        if self.x + self.radius >= self.screen_width:
            self.velocity_x =  -self.velocity_x


        # self.set_position(self.x, self.y)



class PongEnvironment(Env):
    
    def __init__(self):
        super(PongEnvironment, self).__init__()

        self.height = 400
        self.width = 800
        
        self.observation_shape = (self.height, self.width, 3)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)
        
        # so we have only 2 action [0:left, 1:right]
        self.action_space = spaces.Discrete(2)

        self.canvas = np.ones(self.observation_shape) * 1

        # Defining elements present inside the environment
        self.elements = []
        # e.g -> striker, ball ( distinct elements )

    def draw_elements_on_canvas(self):
        
        self.canvas = np.ones(self.observation_shape) * 1

        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y
            self.canvas[y : y + elem_shape[0], x : x + elem_shape[1]] = elem.icon

    def reset(self):
        # episodic return
        self.ep_return = 0

        # Initialize striker, ball object
        self.striker = Striker()
        self.ball = Ball()

        # adding to elements
        self.elements = [self.striker, self.ball]

        self.canvas = np.ones(self.observation_shape) * 1

        # drawing elements on canvas
        self.draw_elements_on_canvas()

        # returning observation
        return self.canvas    

    def has_collided(self, ball):
        # Logic is quite simple, we only care about left-side wall and ball position
        collision = False       
        x, y = ball.get_position()

        # so the window size is height = 400 and width = 800
        # and we only care about y position of ball
        if y <= 0:
            collision = True
        
        if collision == True:
            return True
        return False

    def get_action_meanings(self):
        return {
            0 : "Up",
            1 : "Down"
        }

    def step(self, action):
        # represent current episode is done or not!
        done = False

        # assert!! need to study about this !
        assert self.action_space.contains(action), "Invalid Action!"

        # Reward for executing a step
        reward = 1

        # applying action for striker
        if action == 0:
            self.striker.move("Up")
        elif action == 1:
            self.striker.move("Down")

        
        # now out Ball will spawn
        
        self.ball.move()
        if self.has_collided(self.ball):
            done = True
            reward = -10

        
        self.ep_return += 1
        self.draw_elements_on_canvas()

        return self.canvas, reward, done, []





    def render(self, mode = "human"):
        
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas
            



# env = PongEnvironment()
# obs = env.reset()
# screen = env.render(mode = "rgb_array")
# plt.imshow(screen)

from IPython import display

env = PongEnvironment()
obs = env.reset()


while True:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)

    env.render()

    if done == True:
        break

env.close()
        