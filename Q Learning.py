
# coding: utf-8

# In[1]:


import gym


# In[2]:


env = gym.make("MountainCar-v0")
env.reset()


# In[3]:


done = False
while not done:
    action = 2 # 0 left 1 nothing 2 right
    new_sate,reward, done,_ = env.step(action) #state has position and velocity
    env.render()
env.close()

