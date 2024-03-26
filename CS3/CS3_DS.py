import gymnasium as gym
from gymnasium import spaces 
from stable_baselines3 import SAC
from casadi import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# Import Environment
from RSR_Model_1602 import RSR
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Initialise Env



def rollout_test(Ks, ns, DS):
    reps = 1
    env = RSR(ns,test=True,plot=False,DS = True, DS_uk=DS)
    s,_ = env.reset()
    done = False
    states = []
    actions = []
    rewards = []
    controls = []
    tot_rew = 0
    states = np.zeros([env.Nx,ns,reps])
    actions = np.zeros([env.action_space.low.shape[0],ns,reps])
    rewards = np.zeros([1,reps])
    controls = np.zeros([env.action_space_unnorm.low.shape[0]+1,ns,reps])
    for r_i in range(reps):
        tot_reward = 0
        s,_ = env.reset()
        Ks_i = -1
        i = 0
        a = Ks
        for i in range(ns):
          s, reward, done, _,control = env.step(a)
          states[:,i,r_i] = control['state']
          actions[:,i,r_i] = control['PID_Action']
          tot_reward += reward
          controls[:,i,r_i] = control['control_in']
        rewards[:,r_i] = tot_reward
    tot_rew = np.mean(rewards, axis = 1)
    return states, actions, tot_rew,controls

n = 100

Tc_arr = np.linspace(275,305,n)
D_arr = np.linspace(18.2,19,n)
Ks = np.zeros(15)
CV = np.zeros((n,n))
state_log = np.zeros((13,n,n))
ns = 150
env = RSR(ns,test=True,plot=False)
Fm_j = 18.5
for i in range(n):
  D_i = Tc_arr[i]
  DS = np.array([D_i,Fm_j])
  states, actions, tot_rew,controls = rollout_test(Ks, ns, DS)
  CV[i] = states[10,-1][0]
  state_log[:,i] = states[:,-1][0]
#Create contour plot
np.save('CV.npy',CV)
np.save('states_ds.npy',state_log)

plt.figure()
plt.plot(Tc_arr,CV)
plt.show()
# states = np.load('states_ds.npy')

# CV = np.load('CV.npy')
# fig, ax = plt.subplots(1,figsize=(10,10))
# plt.subplots_adjust(hspace=0.5, wspace=0.5)
# # assuming Tc_arr, D_arr, and states are defined    

# fun = ax.contourf(Tc_arr, D_arr, CV, cmap =  'Spectral_r',alpha=0.8, levels = 20)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes('right', size='5%', pad=0.1)
# fig.colorbar(fun, cax=cax, orientation='vertical')
# ax.set_xlabel('Teperature')
# ax.set_ylabel('Distillate Flow (D)')

# ax.set_title('Mol Fraction of B in Bottoms')


# plt.savefig('DS_states.pdf')
# plt.show()






