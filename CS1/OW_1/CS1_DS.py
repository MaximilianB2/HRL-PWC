import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

Tc_n = 102
Tc_vec = np.linspace(290,445,Tc_n)
ns = 240
env = reactor_class(test = True, ns=ns, DS= True)

CV = np.zeros([Tc_vec.shape[0],2])

for i,Tc in enumerate(Tc_vec):
 
  env.reset()
  for ns_i in range(ns-1):
    s_norm, r, done, info,_  = env.step(Tc)
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
 
  CV[i,:] = env.state[1]
y = np.linspace(1,1,Tc_n)
# print(CV)
SP = [0.85 for i in range(int(Tc_n/3))] + [0.4 for i in range(int(Tc_n/3))] + [0.1 for i in range(int(Tc_n/3))]
plt.figure()
plt.plot(Tc_vec, CV[:,0], color='tab:blue', label = 'Equilibrium Concentration')

# plt.fill_between(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]), y, where=(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) > 21) & (np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) < 28), color='tab:red', alpha=0.4, label="Unstable", edgecolor=None)
# plt.fill_between(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]), y, where=(np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) <= 22) | (np.linspace(0,Tc_vec.shape[0],Tc_vec.shape[0]) >= 27), color='tab:green', alpha=0.4, label="Stable", edgecolor=None)

plt.ylabel('Concentration of species B')
plt.xlabel('Cooling Temperature (K)')
plt.ylim(0,1)
plt.grid(True)
plt.legend()
plt.xlim(np.min(Tc_vec),np.max(Tc_vec))
plt.savefig('CS1_DS.pdf')
plt.show()