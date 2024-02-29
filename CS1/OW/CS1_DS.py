import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


Tc_vec = np.linspace(290,303,15)
env = reactor_class(test = True, ns = 120, DS= True)
ns = 120
CV = np.zeros([Tc_vec.shape[0],2])

for i,Tc in enumerate(Tc_vec):
 
  env.reset()
  for ns_i in range(ns-1):
    s_norm, r, done, info,_  = env.step(Tc)
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
   
  CV[i,:] = s[:2]

print(CV)
plt.subplot(211)
plt.plot(Tc_vec,CV[:,0])
plt.subplot(212)
plt.plot(Tc_vec,CV[:,1])
plt.show()