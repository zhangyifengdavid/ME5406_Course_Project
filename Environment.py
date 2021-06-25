import random
import numpy as np  # To deal with data in form of matrices
import tkinter as tk  # To build GUI
import time  # Time is needed to slow down the agent and to see how he runs

from PIL import Image, ImageTk  # For adding images into the canvas widget
from Parameters import *  # import parameters

# Setting the sizes for the environment
pixels = PIXELS  # pixels
env_height = ENV_HEIGHT  # grid height
env_width = ENV_HEIGHT  # grid width


# Creating class for the environment
class Environment(tk.Tk, object):
    def __init__(self, grid_size):
        super(Environment, self).__init__()
        self.action_space = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.action_space)
        self.n_states = env_width * env_height

        self.title('Frozen lake')
        self.geometry('{0}x{1}'.format(env_height * pixels, env_height * pixels))

        # Dictionaries to draw the final route
        self.d = {}
        self.f = {}

        # Global variable for dictionary with coordinates for the final route
        self.a = {}

        # Key for the dictionaries
        self.i = 0

        # Writing the final dictionary first time
        self.c = True

        # Showing the steps for longest found route
        self.longest = 0

        # Showing the steps for the shortest route
        self.shortest = 0

        # Store the obstacles' position
        self.obstacles_positions = []

        # Store the goal's position
        self.goal_position = None

        # build environment (4x4 or 10x10)
        self.grid_size = grid_size

        self.create_environment()

    def create_environment(self):
        # build 4x4 frozen lake environment
        if self.grid_size == 4:
            self.build_4x4_environment()
            print('Create 4x4 environment!')

        # build 10x10 frozen lake environment
        elif self.grid_size == 10:
            self.build_10x10_environment()
            print('Create 10x10 environment!')
        else:
            print("Please input the correct size(4 or 10)")

    # Function to build the 4x4 grid environment
    def build_4x4_environment(self):
        self.canvas_widget = tk.Canvas(self, bg='white',
                                       height=env_height * pixels,
                                       width=env_width * pixels)

        # Uploading an image for background
        # img_background = Image.open("images/bg.png")
        # self.background = ImageTk.PhotoImage(img_background)
        # # Creating background on the widget
        # self.bg = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.background)

        # Creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')

        # Creating objects of  Obstacles
        # Obstacle type 7 - road closed3
        img_obstacle1 = Image.open("images/road_closed3.png")
        self.obstacle1_object = ImageTk.PhotoImage(img_obstacle1)

        # Creating obstacles themselves
        self.obstacle1 = self.canvas_widget.create_image(pixels * 0, pixels * 3, anchor='nw',
                                                         image=self.obstacle1_object)

        self.obstacle2 = self.canvas_widget.create_image(pixels * 1, pixels * 1, anchor='nw',
                                                         image=self.obstacle1_object)

        self.obstacle3 = self.canvas_widget.create_image(pixels * 3, pixels * 1, anchor='nw',
                                                         image=self.obstacle1_object)

        self.obstacle4 = self.canvas_widget.create_image(pixels * 3, pixels * 2, anchor='nw',
                                                         image=self.obstacle1_object)

        # Final Point
        img_flag = Image.open("images/goal.png")
        self.flag_object = ImageTk.PhotoImage(img_flag)
        self.flag = self.canvas_widget.create_image(pixels * 3, pixels * 3, anchor='nw', image=self.flag_object)

        # Uploading the image of Mobile Robot
        img_robot = Image.open("images/agent3.png")
        self.robot = ImageTk.PhotoImage(img_robot)

        # Creating an agent with photo of Mobile Robot
        self.agent = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.robot)

        # Packing everything
        self.canvas_widget.pack()

        # Record the coordinate of the obstacles/holes
        self.obstacles_positions = [self.canvas_widget.coords(self.obstacle1),
                                    self.canvas_widget.coords(self.obstacle2),
                                    self.canvas_widget.coords(self.obstacle3),
                                    self.canvas_widget.coords(self.obstacle4)]

        self.goal_position = self.canvas_widget.coords(self.flag)

        # Function to build the 10x10 grid environment

    # Function to build the 8x8 grid environment
    def build_10x10_environment(self):
        self.canvas_widget = tk.Canvas(self, bg='white',
                                       height=env_height * pixels,
                                       width=env_width * pixels)

        # Uploading an image for background
        # img_background = Image.open("images/bg.png")
        # self.background = ImageTk.PhotoImage(img_background)
        # # Creating background on the widget
        # self.bg = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.background)

        # Creating grid lines
        for column in range(0, env_width * pixels, pixels):
            x0, y0, x1, y1 = column, 0, column, env_height * pixels
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')
        for row in range(0, env_height * pixels, pixels):
            x0, y0, x1, y1 = 0, row, env_height * pixels, row
            self.canvas_widget.create_line(x0, y0, x1, y1, fill='grey')

        # Creating objects of  Obstacles
        # Obstacle type  - road closed
        img_obstacle1 = Image.open("images/road_closed3.png")
        self.obstacle1_object = ImageTk.PhotoImage(img_obstacle1)

        # Creating obstacles themselves
        self.obstacle1 = self.canvas_widget.create_image(pixels * 0, pixels * 3, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle2 = self.canvas_widget.create_image(pixels * 1, pixels * 1, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle3 = self.canvas_widget.create_image(pixels * 3, pixels * 1, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle4 = self.canvas_widget.create_image(pixels * 5, pixels * 2, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle5 = self.canvas_widget.create_image(pixels * 3, pixels * 4, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle6 = self.canvas_widget.create_image(pixels * 5, pixels * 4, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle7 = self.canvas_widget.create_image(pixels * 6, pixels * 7, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle8 = self.canvas_widget.create_image(pixels * 2, pixels * 7, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle9 = self.canvas_widget.create_image(pixels * 3, pixels * 8, anchor='nw',
                                                         image=self.obstacle1_object)
        self.obstacle10 = self.canvas_widget.create_image(pixels * 8, pixels * 5, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle11 = self.canvas_widget.create_image(pixels * 6, pixels * 6, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle12 = self.canvas_widget.create_image(pixels * 6, pixels * 5, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle13 = self.canvas_widget.create_image(pixels * 5, pixels * 6, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle14 = self.canvas_widget.create_image(pixels * 1, pixels * 6, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle15 = self.canvas_widget.create_image(pixels * 2, pixels * 8, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle16 = self.canvas_widget.create_image(pixels * 8, pixels * 3, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle17 = self.canvas_widget.create_image(pixels * 5, pixels * 8, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle18 = self.canvas_widget.create_image(pixels * 6, pixels * 2, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle19 = self.canvas_widget.create_image(pixels * 8, pixels * 2, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle20 = self.canvas_widget.create_image(pixels * 4, pixels * 1, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle21 = self.canvas_widget.create_image(pixels * 5, pixels * 5, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle22 = self.canvas_widget.create_image(pixels * 8, pixels * 7, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle23 = self.canvas_widget.create_image(pixels * 6, pixels * 1, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle24 = self.canvas_widget.create_image(pixels * 9, pixels * 3, anchor='nw',
                                                          image=self.obstacle1_object)
        self.obstacle25 = self.canvas_widget.create_image(pixels * 3, pixels * 1, anchor='nw',
                                                          image=self.obstacle1_object)

        # Final Point
        img_flag = Image.open("images/goal.png")
        self.flag_object = ImageTk.PhotoImage(img_flag)
        self.flag = self.canvas_widget.create_image(pixels * 9, pixels * 9, anchor='nw', image=self.flag_object)

        # Uploading the image of Mobile Robot
        img_robot = Image.open("images/agent3.png")
        self.robot = ImageTk.PhotoImage(img_robot)

        # Creating an agent with photo of Mobile Robot
        self.agent = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.robot)

        # Packing everything
        self.canvas_widget.pack()

        self.obstacles_positions = [self.canvas_widget.coords(self.obstacle1),
                                    self.canvas_widget.coords(self.obstacle2),
                                    self.canvas_widget.coords(self.obstacle3),
                                    self.canvas_widget.coords(self.obstacle4),
                                    self.canvas_widget.coords(self.obstacle5),
                                    self.canvas_widget.coords(self.obstacle6),
                                    self.canvas_widget.coords(self.obstacle7),
                                    self.canvas_widget.coords(self.obstacle8),
                                    self.canvas_widget.coords(self.obstacle9),
                                    self.canvas_widget.coords(self.obstacle10),
                                    self.canvas_widget.coords(self.obstacle11),
                                    self.canvas_widget.coords(self.obstacle12),
                                    self.canvas_widget.coords(self.obstacle13),
                                    self.canvas_widget.coords(self.obstacle14),
                                    self.canvas_widget.coords(self.obstacle15),
                                    self.canvas_widget.coords(self.obstacle16),
                                    self.canvas_widget.coords(self.obstacle17),
                                    self.canvas_widget.coords(self.obstacle18),
                                    self.canvas_widget.coords(self.obstacle19),
                                    self.canvas_widget.coords(self.obstacle20),
                                    self.canvas_widget.coords(self.obstacle21),
                                    self.canvas_widget.coords(self.obstacle22),
                                    self.canvas_widget.coords(self.obstacle23),
                                    self.canvas_widget.coords(self.obstacle24),
                                    self.canvas_widget.coords(self.obstacle25)]

        self.goal_position = self.canvas_widget.coords(self.flag)

    # Function to reset the environment and start new Episode
    def reset(self):
        self.update()
        # time.sleep(0.1)

        # Updating agent
        self.canvas_widget.delete(self.agent)
        self.agent = self.canvas_widget.create_image(0, 0, anchor='nw', image=self.robot)

        # # Clearing the dictionary and the i
        self.d = {}
        self.i = 0

        # Return observation
        s = self.canvas_widget.coords(self.agent)

        # position transformation(coordinate -> index number)
        s = self.position_transition(s[0], s[1])

        return s

    # Function to get the next observation and reward by doing next step
    def step(self, action):
        # Current state of the agent
        state = self.canvas_widget.coords(self.agent)
        base_action = np.array([0, 0])

        # Updating next state according to the action
        # Action 'up'
        if action == 0:
            if state[1] >= pixels:
                base_action[1] -= pixels
        # Action 'down'
        elif action == 1:
            if state[1] < (env_height - 1) * pixels:
                base_action[1] += pixels
        # Action right
        elif action == 2:
            if state[0] < (env_width - 1) * pixels:
                base_action[0] += pixels
        # Action left
        elif action == 3:
            if state[0] >= pixels:
                base_action[0] -= pixels

        # Moving the agent according to the action
        self.canvas_widget.move(self.agent, base_action[0], base_action[1])

        # Writing in the dictionary coordinates of found route
        self.d[self.i] = self.canvas_widget.coords(self.agent)

        # Updating next state
        next_state = self.d[self.i]

        # Updating key for the dictionary
        self.i += 1

        # Calculating the reward for the agent
        if next_state == self.goal_position:
            reward = 1
            done = True
            # print("reach goal!")
            # next_state = 'goal'

            # Filling the dictionary first time
            if self.c == True:
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]
                self.c = False
                self.longest = len(self.d)
                self.shortest = len(self.d)

            # Checking if the currently found route is shorter
            if len(self.d) < len(self.f):
                # Saving the number of steps for the shortest route
                self.shortest = len(self.d)
                # Clearing the dictionary for the final route
                self.f = {}
                # Reassigning the dictionary
                for j in range(len(self.d)):
                    self.f[j] = self.d[j]

            # Saving the number of steps for the longest route
            if len(self.d) > self.longest:
                self.longest = len(self.d)

        elif next_state in self.obstacles_positions:
            reward = -1
            done = True
            # print("reach obstacle")
            # next_state = 'obstacle'

            # Clearing the dictionary and the i
            self.d = {}
            self.i = 0

        else:
            reward = 0
            done = False

        # position transformation(coordinate -> index number)
        next_state = self.position_transition(next_state[0], next_state[1])

        return next_state, reward, done, {}

    # Function to refresh the environment
    def render(self):
        time.sleep(0.05)
        self.update()

    # Function to show the found route
    def final(self):
        # Deleting the agent at the end
        self.canvas_widget.delete(self.agent)

        # Showing the number of steps
        print('The shortest route:', self.shortest)
        print('The longest route:', self.longest)

        # Creating initial point
        origin = np.array([20, 20])
        self.initial_point = self.canvas_widget.create_oval(
            origin[0] - 5, origin[1] - 5,
            origin[0] + 5, origin[1] + 5,
            fill='blue', outline='blue')

        # Filling the route
        for j in range(len(self.f)):
            # Showing the coordinates of the final route
            self.track = self.canvas_widget.create_oval(
                self.f[j][0] + origin[0] - 5, self.f[j][1] + origin[0] - 5,
                self.f[j][0] + origin[0] + 5, self.f[j][1] + origin[0] + 5,
                fill='blue', outline='blue')
            # Writing the final route in the global variable a
            self.a[j] = self.f[j]

    # Returning the final dictionary with route coordinates
    # Then it will be used in agent_brain.py
    def final_states(self):
        return self.a

    def position_transition(self, x, y):
        width = self.grid_size
        # Coordinate transformation: Coordinate-> Indexed number
        s = int(x / 40) + int(y / 40 * width)
        return s


def update():
    for t in range(100):
        s = env.reset()
        while True:
            env.render()
            a = random.randint(0, 3)
            s_, r, done, info = env.step(a)
            if done:
                break


# If we want to run and see the environment without running full algorithm
if __name__ == '__main__':
    # Create environment
    env = Environment(grid_size=GRID_SIZE)

    # update()
    env.mainloop()
