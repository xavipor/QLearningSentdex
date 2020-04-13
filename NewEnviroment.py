# -*- coding: utf-8 -*-
"""
Created on Sun May 19 17:55:53 2019

@author: Javier Dominguez
"""
import numpy as np  # for array stuff and random
from PIL import Image  # for creating visual of our env
import cv2  # for showing our visual live
import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import pickle  # to save/load Q-Tables
from matplotlib import style  # to make pretty charts because it matters.
import time  # using this to keep track of our saved Q-Tables.

style.use("ggplot")  # setting our style!

SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1  # feel free to tinker with these!
ENEMY_PENALTY = 300  # feel free to tinker with these!
FOOD_REWARD = 25  # feel free to tinker with these!
epsilon = 0.5  # randomness
EPS_DECAY = 0.9999  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1000  # how often to play through env visually.

start_q_table = None  # if we have a pickled Q table, we'll put the filename of it here.

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

# the dict! Using just for colors
d = {1: (255, 175, 0),  # blueish color
     2: (0, 255, 0),  # green
     3: (0, 0, 255)}  # red


class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"
    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)     
    
    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
    
    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1

if start_q_table is None:
    q_table={}
    for x1 in range (-SIZE+1,SIZE):
        for y1 in range (-SIZE+1,SIZE):
            for x2 in range (-SIZE+1,SIZE):
                for y2 in range (-SIZE+1,SIZE):
                    q_table[(x1,y1),(x2,y2)]=[np.random.uniform(-5,0) for i in range(4)] #tenemos 4 acciones 
else:
    with open(start_q_table,"rb") as f:
        q_talbe = pickle.load(f)
        
episode_rewards = []        
for episode in range (HM_EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    
    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}" )
        show = True
    else:
        show = False
        
    episode_reward = 0
    for i in range(200):
        obs = (player-food,player-enemy)
        if np.random.random()>epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)
        
        player.action(action)
        '''
         #si quremos que se mueva el enemigo y la comida...
        enemy.move()
        food.move
        '''  
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
            
        new_obs = (player - food, player - enemy)
        max_future_q = np.max(q_table[obs])
        current_q = q_table[obs][action]
        
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD #cuando la tenemos tya esta
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1 - LEARNING_RATE) *  current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q
        
        if show:
            env = np.zeros((SIZE,SIZE,3),dtype = np.uint8)#all black enviroment
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]
            
            img = Image.fromarray(env,"RGB")
            img = img.resize((300,300))
            cv2.imshow("",np.array(img))
            if reward = FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            
        
                    