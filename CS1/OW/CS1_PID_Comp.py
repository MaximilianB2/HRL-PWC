import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import differential_evolution

ns = 240
reps = 10
Ca_des = [0.95 for i in range(int(ns/3))] + [0.9 for i in range(int(ns/3))] + [0.85 for i in range(int(ns/3))]         
T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]



def rollout(Ks, PID_Form,opt,reps):
  ns = 240
 
  Ca_des = [0.95 for i in range(int(ns/3))] + [0.9 for i in range(int(ns/3))] + [0.85 for i in range(int(ns/3))]  
  T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]
  Ca_eval = np.zeros((ns,reps))
  T_eval = np.zeros((ns,reps))
  Tc_eval = np.zeros((ns,reps))
  ks_eval = np.zeros((3,ns,reps))
  r_eval = np.zeros((1,reps))
  SP = np.array([Ca_des,T_des])
  x_norm = np.array(([-200,0,0.01],[0,20,10]))
  if PID_Form == 'pos':
    env = reactor_class(test = True, ns = 240, PID_pos = True)
  elif PID_Form == 'vel':
    env = reactor_class(test = True, ns = 240, PID_vel = True)
  
  for r_i in range(reps):
    s_norm,_ = env.reset()
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval[0,r_i] = s[0]
    T_eval[0,r_i] = s[1]
    Tc_eval[0,r_i] = 300.0
    Ks_norm = ((Ks[:3] + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
    ks_eval[:,0,r_i] = Ks_norm
    r_tot = 0
    Ks_i = 0
    for i in range(1,ns):
      if i % 5 == 0:
        Ks_i += 1
      s_norm, r, done, _,info = env.step(Ks[Ks_i*3:(Ks_i+1)*3])
      
      ks_eval[:,i,r_i] = info['Ks']
      r_tot += r
      s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
      Ca_eval[i,r_i] = s[0]
      T_eval[i,r_i] = s[1]
      Tc_eval[i,r_i] = env.u_history[-1]
    r_eval[:,r_i] = r_tot
  r = -1*np.mean(r_tot)
  

  if opt:
    return r
  else:
    print(r)
    return Ca_eval, T_eval, Tc_eval, ks_eval


def plot_simulation_comp(Ca_dat_PG, T_dat_PG, Tc_dat_PG,ks_eval_PG,SP,ns):
  plt.rcParams['text.usetex'] = 'False'
  t = np.linspace(0,25,ns)
  fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  labels = ['$Ca_{k_p}$','$Ca_{k_i}$','$Ca_{k_d}$','$T_{k_p}$','$T_{k_i}$','$T_{k_d}$']
  col = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']
  col_fill = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']

  axs[0].plot(t, np.median(Ca_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'velocity form')
  axs[0].fill_between(t, np.min(Ca_dat_PG,axis=1), np.max(Ca_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')

  axs[0].step(t, Ca_des, '--', lw=1.5, color='black')
  axs[0].set_ylabel('Ca (mol/m$^3$)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))

  axs[1].plot(t, np.median(T_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'velocity form')
  axs[1].fill_between(t, np.min(T_dat_PG,axis=1), np.max(T_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')

  axs[1].set_ylabel('Temperature (K))')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))
   
  axs[2].step(t, np.median(Tc_dat_PG,axis=1), 'r--', where = 'post',lw=1, label = 'velocity form')
  axs[2].set_ylabel('Cooling T (K)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))
  # plt.savefig('velocity_vs_pos_states.pdf')
  plt.show()

 
  fig, axs = plt.subplots(2, 1, figsize=(10, 12))
  axs[0].set_title('velocity form PID Parameters')

  axs[0].step(t, np.median(ks_eval_PG[0,:,:],axis=1), col[0],where = 'post', lw=1,label = labels[0])
    # plt.gca().fill_between(t, np.min(ks_eval_PG[ks_i,:,:],axis=1), np.max(ks_eval_PG[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)
                            
  axs[0].set_ylabel('Ca PID Parameter (velocity form)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))
  for ks_i in range(1,3):
    axs[1].step(t, np.median(ks_eval_PG[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = labels[ks_i])
    # plt.gca().fill_between(t, np.min(ks_eval_PG[ks_i,:,:],axis=1), np.max(ks_eval_PG[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)
                            
  axs[1].set_ylabel('Ca PID Parameter (velocity form)')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))
  plt.tight_layout()
  # plt.savefig('velocity_vs_pos_ks_vel.pdf')
  plt.show()



bounds = [(-1,1)]*3*48
x0 = np.array([-157,1.2,0.01]*48).reshape((48,3))
x_norm = np.array(([-200,0,0.01],[0,20,10]))


x0_norm = (((x0-x_norm[0]) / (x_norm[1] - x_norm[0]))*2 - 1).flatten()



#x0_norm = np.tile(x0_norm, (popsize, 1))

# def save_current_state(intermediate_result):
#     np.save('current_solution.npy', intermediate_result.x)
#     with open('current_function_value.txt', 'w') as f:
#         f.write(str(intermediate_result.fun))
# print('Starting Velocity Opt')
result_vel =  differential_evolution(rollout, polish= False,popsize = 3, bounds=bounds, args= ('vel', True,3), maxiter = 5000,disp = True)
np.save('GS_Global_vel.npy', result_vel.x)
# Ks_vel = result_vel.x
# Ks_vel = np.load('GS_Global_vel.npy')
# np.save('GS_Global_vel_const.npy',Ks_vel)
# Ks_vel = np.load('GS_Global_vel_reducedPop.npy')
# Ks_pos = np.load('GS_Global_pos_const.npy')
# Ks_vel = np.load('GS_Global_vel_const.npy')
# Ks_pos = np.load('GS_Global_pos_0603.npy')
# print(Ks_vel)

# print(Ks_vel)
# # Ks_vel = Ks_pos =  np.array([-0.5,-0.8,-1,0.8]*48)

# SP = np.array([Ca_des,T_des])
# Ca_dat_vel, T_dat_vel, Tc_dat_vel, ks_eval_vel = rollout(Ks_vel, 'vel', opt = False,reps = 10)

# plot_simulation_comp(Ca_dat_vel, T_dat_vel, Tc_dat_vel,ks_eval_vel,SP,ns)

