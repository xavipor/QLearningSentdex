#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import style


# In[2]:


env = gym.make("MountainCar-v0")
env.reset()


# In[13]:


print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

LEARNING_RATE = 0.1
DISCOUNT = 0.95 #measute of how important we see future actions. Future rewards vs current reward... 0.95*0.95*0.95...
EPISODES = 10000
SHOW_EVERY = 500
STATS_EVERY=100

DISCRETE_SIZE_Q_TABLE = [40] * len(env.observation_space.high) #lo hacemos así para tener un tamño 
discrete_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_SIZE_Q_TABLE

epsilon = 0.5 #Cuanto mas grande  mas probable que hagamos una acción random (exploracion). Por si nos atascamos
START_EPSYLON_DECAYING = 1
END_EPSYLON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon/(END_EPSYLON_DECAYING - START_EPSYLON_DECAYING) #Cuanto valor queremos hacer el decay 

q_table = np.random.uniform(low = -2, high=0,size =(DISCRETE_SIZE_Q_TABLE + [env.action_space.n]) )

ep_rewards = []
aggr_ep_rewards = {'ep':[], 'avg':[],'min':[],'max':[]}


#la historia es que el enviroment nos lo da en continio y nosotros tenemos espacios discretos de posición y velocidad
def get_discrete_state (state):
    discrete_state = (state - env.observation_space.low)/discrete_size
    return tuple(discrete_state.astype(np.int))





#decente con el que trabajar

print(discrete_size)

#El tamaño requiere de el tamaño de estados y "una matriz" por cada acción
print(q_table.shape) #50x50x3



# In[20]:



for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())#para conseguir el initial state
    done = False
    episode_reward = 0
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    
    
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state]) # 0 left 1 nothing 2 right
        else:#random action
            action = np.random.randint(0, env.action_space.n)
            
        new_state,reward, done,_ = env.step(action) #state has position and velocity
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE)* current_q + LEARNING_RATE * (reward + DISCOUNT* max_future_q)
            q_table[discrete_state+(action,)] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state
        #print(reawar,new_state)
    if END_EPSYLON_DECAYING >= episode >= START_EPSYLON_DECAYING:
        epsilon -= epsilon_decay_value
        
    ep_rewards.append(episode_reward)
    if not episode % STATS_EVERY:
        average_reward = sum(ep_rewards [-STATS_EVERY:])/STATS_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards [-STATS_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards [-STATS_EVERY:]))
        print(f"Episode: {episode} avg: {average_reward},min: {min(ep_rewards [-SHOW_EVERY:])}, max {max(ep_rewards [-SHOW_EVERY:])}" )
        np.save(f"{episode}-qtable.npy", q_table)
        

        
env.close()
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['avg'],label = 'avg')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['min'],label = 'min')
plt.plot(aggr_ep_rewards['ep'],aggr_ep_rewards['max'],label = 'max')
plt.legend(loc=4)
plt.show()
#with this we have no power to get to the top of the mountain. we need to get momentum


# In[22]:


style.use('ggplot')

def get_q_color(value,vals):
    if value == max(vals):
        return "green", 1.0
    else:
        return "red", 0.3

fig = plt.figure(figsize=(12,9))

for i in range(0,10000,100):
    print(i)
    ax1 = fig.add_subplot(311)
    ax2 = fig.add_subplot(312)
    ax3 = fig.add_subplot(313)
    q_table = np.load(f"{i}-qtable.npy")
    
    
    counter = 0
    for x, x_vals in enumerate(q_table):
        if counter == 0:
            print(x,x_vals)
        for y, y_vals in enumerate(x_vals): #separamos la discretización de 0 49
            if counter == 0:
                print(y,y_vals)
                counter =1
            ax1.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax2.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            ax3.scatter(x, y, c=get_q_color(y_vals[0], y_vals)[0], marker="o", alpha=get_q_color(y_vals[0], y_vals)[1])
            
            ax1.set_ylabel("Action 0")
            ax2.set_ylabel("Action 1")
            ax3.set_ylabel("Action 2")
    plt.savefig("f{i}.png")
    plt.clf()


# In[ ]:
import cv2
import os
def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    for i in range(0:10000:100):
        img_path = f"{i}.png"
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()
    
make_video()


