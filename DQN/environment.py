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
        self.x = 10
        self.y = 200
        self.width = 15
        self.height = 60
        self.y_min = 0
        self.y_max = 400
        self.velocity = 5
        self.screen_height = 400

        # now creating icon for it
        self.icon = cv2.imread("DQN/paddle.png") / 255.0
        # self.icon = cv2.imread("paddle.png") / 255
        self.icon = cv2.resize(self.icon, (self.width, self.height))
        
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
             
    
class Ball:
    def __init__(self):
        self.x = 400
        self.y = 200
        self.width = 10
        self.height = 10
        self.velocity_x = 4
        self.velocity_y = 4
        self.color = (255, 255, 255)
        self.screen_width = 800
        self.screen_height = 400

        self.icon = cv2.imread("DQN/ball.png") / 255.0
        self.icon = cv2.resize(self.icon, (self.width, self.height))

    def get_position(self):
        return (self.x, self.y)

    def move(self):
        self.x -= self.velocity_x
        self.y -= self.velocity_y

        # reflect from TOP and BOTTOM 
        if self.y <= 3 or self.y + self.height >= self.screen_height - 3:
            self.velocity_y = -self.velocity_y
        # reflect from right wall
        if self.x + self.width >= self.screen_width - 3:
            self.velocity_x = -self.velocity_x

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
        self.reward = 0

    def draw_elements_on_canvas(self):
        
        self.canvas = np.ones(self.observation_shape) * 1

        for elem in self.elements:
            elem_shape = elem.icon.shape
            x, y = elem.x, elem.y

            self.canvas[y : y + elem_shape[0], x : x + elem_shape[1]] = elem.icon

        text = 'state : {}, Rewards : {}'.format(self.ep_return, self.reward)
        self.canvas = cv2.putText(self.canvas, text, (10, 20), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)    
    
    def reset(self):
        # episodic return
        self.ep_return = 0
        self.reward = 0

        # Initialize striker, ball object
        self.striker = Striker()
        self.ball = Ball()

        # adding to elements
        self.elements = [self.striker, self.ball]

        self.canvas = np.ones(self.observation_shape) * 1

        # drawing elements on canvas
        self.draw_elements_on_canvas()

    def has_collided(self, ball):
        # Logic is quite simple, we only care about left-side wall and ball position
        collision = False       
        x, y = ball.get_position()

        # so the window size is height = 400 and width = 800
        # and we only care about y position of ball
        if x <= 3:
            collision = True
            return True
        return False
    
    def striker_collision(self, striker, ball):
        x_collision = False
        y_collision = False

        ball_x, ball_y = ball.get_position()
        striker_x, striker_y = striker.get_position()
        # print(ball_x, ball_y)


        if ball_x < striker_x + striker.width and ball_x + ball.width > striker_x:
            x_collision = True
        if ball_y < striker_y + striker.height and ball_y + ball.height > striker_y:
            y_collision = True
        
        return x_collision and y_collision

    def get_action_meanings(self):
        return {
            0 : "Up",
            1 : "Down"
        }

    def step(self, action, skip_steps=4):
        total_reward = 0
        done = False
        info = {}

        for _ in range(skip_steps):
            assert self.action_space.contains(action), "Invalid Action!"

            # applying action for striker
            if action == 0:
                self.striker.move("Up")
            elif action == 1:
                self.striker.move("Down")

            self.ball.move()

            # checking if ball is collided with striker
            if self.striker_collision(self.striker, self.ball):
                done = True
                self.ball.velocity_x = -self.ball.velocity_x
                total_reward += 1
            

            # checking if ball is collided with left wall
            if self.has_collided(self.ball):
                done = True
                total_reward -= 1
                break  # End the episode if the ball collides with the left wall

            self.ep_return += 1
            self.draw_elements_on_canvas()

            if done:
                break

        # state = [self.striker.x, self.striker.y, self.ball.x, self.ball.y]
        return [], total_reward, done, info


    def get_striker_and_ball_coordinates(self):
        return [self.striker.x, self.striker.y, self.ball.x, self.ball.y, self.ball.velocity_x, self.ball.velocity_y]

    def render(self, mode = "human"):
        
        assert mode in ["human", "rgb_array"], "Invalid mode, must be either \"human\" or \"rgb_array\""
        if mode == "human":
            cv2.imshow("game", self.canvas)
            cv2.waitKey(10)

        elif mode == "rgb_array":
            return self.canvas
            

# from IPython import display
# display.clear_output(wait=True)

# env = PongEnvironment()
# obs = env.reset()  


# while True:
#     action = env.action_space.sample()
#     obs, reward, done, info = env.step(action)

#     env.render()

#     if done == True: 
#         # break
#         obs = env.reset()
# # env.close()
        
