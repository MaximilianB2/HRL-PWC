import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize
ns = 240
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
      if PID_Form == 'Const':
       if i % 240 == 0:
          Ks_i += 1
      elif PID_Form == 'GS':
        if i % 80 == 0:
          Ks_i += 1
      s_norm, r, done, _,info = env.step(Ks[Ks_i*3:(Ks_i+1)*3])
      
      ks_eval[:,i,r_i] = info['Ks']
      r_tot += r
      s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
      Ca_eval[i,r_i] = s[0]
      T_eval[i,r_i] = s[1]
      Tc_eval[i,r_i] = env.u_history[-1]
    r_eval[:,r_i] = r_tot
 
  r = -1*np.mean(r_eval,axis=1)

  ISE = np.sum((Ca_des - np.median(Ca_eval,axis=1))**2)
  
  if opt:
    return r
  else:
    print(r)
    print(ISE,'ISE')
    return Ca_eval, T_eval, Tc_eval, ks_eval



def plot_simulation_comp(Ca_dat_PG, T_dat_PG, Tc_dat_PG,ks_eval_PG,Ca_dat_const, T_dat_const, Tc_dat_const,ks_eval_const,SP,ns):
  plt.rcParams['text.usetex'] = 'False'
  t = np.linspace(0,25,ns)
  fig, axs =  plt.subplots(1,3,figsize=(20, 7))
  labels = ['$Ca_{k_p}$','$Ca_{k_i}$','$Ca_{k_d}$','$T_{k_p}$','$T_{k_i}$','$T_{k_d}$']
  col = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']
  col_fill = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']

  axs[0].plot(t, np.median(Ca_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'Gain Schedule')
  axs[0].fill_between(t, np.min(Ca_dat_PG,axis=1), np.max(Ca_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')
  axs[0].plot(t, np.median(Ca_dat_const,axis=1), color = 'tab:blue', lw=1, label = 'Constant Ks')
  axs[0].fill_between(t, np.min(Ca_dat_const,axis=1), np.max(Ca_dat_const,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
  axs[0].step(t, Ca_des, '--', lw=1.5, color='black')
  axs[0].set_ylabel('Ca (mol/m$^3$)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))

  axs[1].plot(t, np.median(T_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'Gain Schedule')
  axs[1].fill_between(t, np.min(T_dat_PG,axis=1), np.max(T_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')

  axs[1].plot(t, np.median(T_dat_const,axis=1), color = 'tab:blue', lw=1, label = 'Constant Ks')
  axs[1].fill_between(t, np.min(T_dat_const,axis=1), np.max(T_dat_const,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
  axs[1].set_ylabel('Temperature (K))')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))
   
  axs[2].step(t, np.median(Tc_dat_PG,axis=1), color = 'tab:red', linestyle = 'dashed', where = 'post',lw=1, label = 'Gain Schedule')
  axs[2].step(t, np.median(Tc_dat_const,axis=1), color = 'tab:blue', linestyle = 'dashed', where = 'post',lw=1, label = 'Constant Ks')
  axs[2].set_ylabel('Cooling T (K)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))
  plt.savefig('gs_vs_const_states.pdf')
  plt.show()

 
  fig, axs =  plt.subplots(1,3,figsize=(20, 7))
  axs[0].set_title('Gain Schedule PID Parameters')

  axs[0].step(t, np.median(ks_eval_PG[0,:,:],axis=1), col[0],where = 'post', lw=1,label = 'GS ' + labels[0])
  axs[0].fill_between(t, np.min(ks_eval_PG[0,:,:],axis=1), np.max(ks_eval_PG[0,:,:],axis=1),color=col_fill[0], alpha=0.2)
  axs[0].step(t, np.median(ks_eval_const[0,:,:],axis=1), color = 'black',where = 'post', lw=1,label = 'Constant Ks ' + labels[0])
  axs[0].fill_between(t, np.min(ks_eval_const[0,:,:],axis=1), np.max(ks_eval_const[0,:,:],axis=1),color='black', alpha=0.2)
                            
  axs[0].set_ylabel('Ca PID Parameter (Gain Schedule)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))
  for ks_i in range(1,3):
    axs[ks_i].step(t, np.median(ks_eval_PG[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = 'GS ' + labels[ks_i])
    axs[ks_i].fill_between(t, np.min(ks_eval_PG[ks_i,:,:],axis=1), np.max(ks_eval_PG[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)
    axs[ks_i].step(t, np.median(ks_eval_const[ks_i,:,:],axis=1), 'black', where = 'post', lw=1,label = 'Constant Ks ' + labels[ks_i])
    axs[ks_i].fill_between(t, np.min(ks_eval_const[ks_i,:,:],axis=1), np.max(ks_eval_const[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)                        
    axs[ks_i].set_ylabel('Ca PID Parameter (Gain Schedule)')
    axs[ks_i].set_xlabel('Time (min)')
    axs[ks_i].legend(loc='best')
    axs[ks_i].set_xlim(min(t), max(t))
  plt.tight_layout()
  plt.savefig('gs_vs_const_ks_vel.pdf')
  plt.show()
SP = np.array([Ca_des,T_des])

Ks_GS = np.load('GS_Global_vel.npy')
Ca_dat_GS, T_dat_GS, Tc_dat_GS, ks_eval_GS = rollout(Ks_GS, 'GS', opt = False,reps = 10)

Ks_const = np.load('GS_Global_vel_const.npy')
Ca_dat_const, T_dat_const, Tc_dat_const, ks_eval_const = rollout(Ks_const, 'const', opt = False,reps = 10)
plot_simulation_comp(Ca_dat_GS, T_dat_GS, Tc_dat_GS,ks_eval_GS,Ca_dat_const, T_dat_const, Tc_dat_const,ks_eval_const,SP,ns)