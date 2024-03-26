import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

Tc_n = 102
Tc_vec = np.linspace(290,304,Tc_n)
ns = 240
env = reactor_class(test = True, ns=ns, DS= True)

CV = np.zeros([Tc_vec.shape[0],2])

for i,Tc in enumerate(Tc_vec):
 
  env.reset()
  for ns_i in range(ns-1):
    s_norm, r, done, info,_  = env.step(Tc)
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
   
  CV[i,:] = s[:2]
y = np.linspace(1,1,Tc_n)
# print(CV)
SP = [0.85 for i in range(int(Tc_n/3))] + [0.4 for i in range(int(Tc_n/3))] + [0.1 for i in range(int(Tc_n/3))]
plt.subplot(211)
plt.plot(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]), CV[:,0], color='tab:blue', label = 'Equilibrium Concentration')
plt.step(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]), SP ,color = 'black',linestyle= '--',alpha = 0.5,label ='Proposed Setpoints')
# plt.fill_between(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]), y, where=(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) > 21) & (np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) < 28), color='tab:red', alpha=0.4, label="Unstable", edgecolor=None)
# plt.fill_between(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]), y, where=(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) <= 22) | (np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) >= 27), color='tab:green', alpha=0.4, label="Stable", edgecolor=None)

plt.ylabel('Concentration of species A')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.xlim(np.min(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0])),np.max(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0])))
plt.subplot(212)
plt.step(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]),Tc_vec,color = 'tab:red')
plt.grid(True)
plt.ylabel('Cooling Temperature')
plt.xlim(np.min(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0])),np.max(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0])))
plt.show()