import numpy as np
import os
from random import randint
import cv2 as cv
from numpy import argmax



class GameGrid(object):
    """sets up the environment"""

    def __init__(self, field_size):
        """initializes the environment"""

        self.field_size = field_size
        # self.total_walls = 2*self.field_size
        # self.total_deeppits = self.field_size/10
        # self.total_shallowpits = self.field_size/10
        self.reset()


    def reset(self):
        """resets the environment"""


        self.player = (0,0)
        self.deep_pits = [(0,0)]
        self.shallow_pits = [(0,0)]
        self.walls = [(0,0)]
        self.goal = (0,0)


        self.init_grid()
        #self.createImage()
        #self.setup_grid()
        #return self.state


    def init_grid(self):

        """initializes elements in the environment"""

        # goal --> top right of grid
        # (x,y) coordinates based on numpy 2d array indexing

        self.goal = [(0,self.field_size-1)]

        self.walls = self.assign_walls()
        self.deep_pits, self.shallow_pits = self.assign_pits()
        #self.pellet = self.assign_pellet()
        self.grid = self.setup_grid()
        self.player_start_position()
        self.timer = 0
        self.power = False


    def assign_walls(self):
        """assigns walls on the grid"""

        # (x,y) coordinates based on numpy 2d array indexing
        self.walls = [(3,0),(6,0),(1,1),(2,1),(3,1),(6,1),(7,1),(8,1),(15,1),(3,2),(6,2),(9,2),(15,2),(16,2),(17,2),
                            (6,3),(9,3),(10,3),(11,3),(15,3),(4,4),(5,4),(6,4),(9,4),(15,4),(6,5),(12,5),(13,5),(14,5),(15,5),(4,6),
                            (10,6),(11,6),(12,6),(15,6),(4,7),(5,7),(6,7),(12,7),(4,8),(4,9),(8,9),(11,9),(18,9),(3,10),(4,10),(5,10),
                            (6,10),(8,10),(9,10),(10,10),(11,10),(12,10),(13,10),(16,10),(17,10),(18,10),(1,11),(2,11),(3,11),(4,11),
                            (8,11),(11,11),(18,11),(3,12),(11,12),(6,13),(11,13),(12,13),(13,13),(4,14),(5,14),(6,14),(11,14),(6,15),
                            (17,15),(2,16),(7,16),(10,16),(17,16),(18,16),(19,16),(2,17),(3,17),(4,17),(5,17),(6,17),(7,17),(8,17),(9,17),
                            (10,17),(15,17),(17,17),(2,18),(7,18),(10,18),(13,18),(14,18),(15,18),(15,19)]

        return self.walls

    def assign_pits(self):
        """assigns pits on the grid"""

        # (x,y) coordinates in a regular x,y plane
        # self.deep_pits = [(9,19),(19,11)]
        # self.shallow_pits = [(6,11),(11,4)]

        # (x,y) coordinates based on numpy 2d array indexing
        self.deep_pits = [(0,9),(7,19),(12,1),(19,11)]
        self.shallow_pits = [(5,5),(15,11),(7,10),(17,5)]

        return self.deep_pits, self.shallow_pits


    def setup_grid(self):
        """sets up elements in the environment"""

        # Player = 1
        # Wall = 2
        # Shallow Pit = 3
        # Deep Pit = 4
        # Goal = 5

        self.grid = np.zeros((self.field_size, self.field_size))


        for goal_position in self.goal:
            self.grid[goal_position[0], goal_position[1]] = 5

        for wall_position in self.walls:
            self.grid[wall_position[0], wall_position[1]] = 2

        for shallow_pit in self.shallow_pits:
            self.grid[shallow_pit[0], shallow_pit[1]] = 3

        for deep_pit in self.deep_pits:
            self.grid[deep_pit[0], deep_pit[1]] = 4

        return self.grid



    def player_start_position(self):
        """randomizes the start position of player in each episode"""

        invalidStartPosition = True
        while invalidStartPosition:
            x_coordinate = randint(0, self.field_size-1)
            y_coordinate = randint(0, self.field_size-1)

            if self.grid[x_coordinate,y_coordinate] == 0:
                invalidStartPosition = False
                self.player = (x_coordinate, y_coordinate)
            else:
                continue

        #self.player = (2,5)
        self.player_start = self.player
        self.grid[self.player[0], self.player[1]] = 1



    def update_player(self):
        """updates player position in the environment"""
        self.grid[self.player[0], self.player[1]] = 1


    def createImage(self):
        """creates image of environment state"""

        if not os.path.exists(os.path.join(os.getcwd(),'grid.jpg')):

            grid_img = np.zeros([200,200,3])

            grid_img[:,:,0] = np.ones([200,200])*255
            grid_img[:,:,1] = np.ones([200,200])*255
            grid_img[:,:,2] = np.ones([200,200])*255


            cv.imwrite('grid.jpg',grid_img)

        img = cv.imread('grid.jpg')

        for i in range(20):
            for j in range(20):

                if self.grid[i,j] == 0:
                    cv.rectangle(img, (j*10,i*10), (j*10 +10 ,i*10 +10), (255,178,102), thickness= -1)
                if self.grid[i,j] == 1:
                    cv.rectangle(img, (j*10,i*10), (j*10 +10 ,i*10 +10), (0,255,255), thickness= -1)
                if self.grid[i,j] == 2:
                    cv.rectangle(img, (j*10,i*10), (j*10 +10 ,i*10 +10), (153,0,0), thickness= -1)
                if self.grid[i,j] == 3:
                    cv.rectangle(img, (j*10,i*10), (j*10 +10 ,i*10 +10), (51,153,255), thickness= -1)
                if self.grid[i,j] == 4:
                    cv.rectangle(img, (j*10,i*10), (j*10 +10 ,i*10 +10), (0,0,255), thickness= -1)
                if self.grid[i,j] == 5:
                    cv.rectangle(img, (j*10,i*10), (j*10 +10 ,i*10 +10), (0,204,0 ), thickness= -1)
                if self.grid[i,j] == 6:
                    cv.rectangle(img, (j*10,i*10), (j*10 +10 ,i*10 +10), (255,255,255 ), thickness= -1)


        color_frame = cv.resize(img, (200,200))
        grayscale_frame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        #cv.imshow('img',cv.cvtColor(grid2,cv.COLOR_BGR2GRAY))
        #cv.imshow('img',cv.resize(img, (800,800)))
        #cv.waitKey(0)
        return grayscale_frame, color_frame


    def preprocess_frame(self,frame):
        """normalizes pixels of the image and resizes it"""


#         cv.imshow('img',frame_gray)
#         cv.waitKey(0)

        # Normalize Pixel Values
        normalized_frame = frame/255.0

        # Resize
        preprocessed_frame = cv.resize(normalized_frame, (84,84))

        return preprocessed_frame



    def perform_action(self, command):
        """player performs action in the environment"""

        action = np.argmax(command)
        old_loc = self.player

        # Up (in numpy matrix: x coordinate = x - 1)
        if action == 0:
            if self.player[0] > 0:
                self.player = (self.player[0] - 1, self.player[1])
            else :
                self.player = old_loc

        # Down (in numpy matrix: x coordinate = x + 1)
        elif action == 1:
            if self.player[0] < self.field_size - 1:
                self.player = (self.player[0] + 1, self.player[1])
            else :
                self.player = old_loc

        # Left (in numpy matrix: y coordinate = y - 1)
        elif action == 2:
            if self.player[1] > 0:
                self.player = (self.player[0], self.player[1] - 1)
            else :
                self.player = old_loc

        # Right (in numpy matrix: y coordinate = y + 1)
        elif action == 3:
            if self.player[1] < self.field_size - 1:
                self.player = (self.player[0], self.player[1] + 1)
            else :
                self.player = old_loc


        if self.player in self.deep_pits: # player walked into a deep pit, end episode
            reward = -100
            terminal = True


        elif self.player in self.shallow_pits: # player walked into a shallow pit
            reward = -0.3
            terminal = False

        elif self.player in self.goal: # player walked into goal, end episode
            reward = 100
            terminal = True

        elif self.player in self.walls: # player walked into a wall
            self.player = old_loc  # player did not move
            reward = -0.2
            terminal = False

        else:
            reward = -0.1
            terminal = False

        self.setup_grid()
        self.update_player()
        #self.createImage()

        return terminal,reward

    def getStartLoc(self):
        return self.player_start

    def setStartLoc(self,start):
        self.player= start

#game =GameGrid(20)
#game.createImage()
