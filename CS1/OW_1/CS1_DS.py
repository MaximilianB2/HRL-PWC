import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1 import make_axes_locatable
n = 20
Tc_vec = np.linspace(290,450,n)
F_vec = np.linspace(99,103,n)
ns = 120
env = reactor_class(test = True, ns=ns, DS= True)

x_b = np.zeros((n,n))
V = np.zeros_like(x_b)
# for j, F_j in enumerate(F_vec):
#   for i,Tc in enumerate(Tc_vec):
#     env.reset()
#     for ns_i in range(ns-1):
#       u = np.array([Tc,F_j])
#       s_norm, r, done, info,_  = env.step(u)
#       s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low

#     x_b[i,j] = env.state[1]
#     x_b[x_b < 0] = 0
#     V[i,j] = env.state[4]

# fig, ax = plt.subplots(2,figsize=(10,10))
# plt.subplots_adjust(hspace=0.5, wspace=0.5)

# fun = ax[0].contourf(F_vec, Tc_vec, x_b, cmap='Spectral_r', alpha=0.8)
# divider = make_axes_locatable(ax[0])
# cax = divider.append_axes('right', size='5%', pad=0.1)
# cbar = fig.colorbar(fun, cax=cax, orientation='vertical')
# ax[0].set_ylabel('Temperature')
# ax[0].set_xlabel('Flow in')
# ax[0].set_title('Mol Fraction of B')

# fun = ax[1].contourf(F_vec, Tc_vec, V, cmap='Spectral_r', alpha=0.8)
# divider = make_axes_locatable(ax[1])
# cax = divider.append_axes('right', size='5%', pad=0.1)
# fig.colorbar(fun, cax=cax, orientation='vertical')
# ax[1].set_ylabel('Temperature')
# ax[1].set_xlabel('Flow in')
# plt.savefig('CS1_2x2_DS.pdf')
# plt.show()



F_j = 100
x_b_slice = np.zeros(n)
Tc_vec = np.linspace(290,450,n)
for i,Tc in enumerate(Tc_vec):
  env.reset()
  for ns_i in range(ns-1):
    u = np.array([Tc,F_j])
    s_norm, r, done, info,_  = env.step(u)
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
  x_b_slice[i] = env.state[1]
  x_b_slice[x_b_slice < 0] = 0




plt.figure()
plt.plot(Tc_vec,x_b_slice,color = 'tab:blue')
plt.xlim(np.min(Tc_vec),np.max(Tc_vec))
plt.xlabel('Cooling Temperature')
plt.ylabel('Concentration of species B')
plt.grid(True)
plt.title('Slice at F = 100m3/min')
plt.savefig('cs1_2x2_slice.pdf')
plt.show()