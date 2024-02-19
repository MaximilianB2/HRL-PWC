import os
from stable_baselines3 import SAC
import numpy as np
import matplotlib.pyplot as plt
from CS1_Model import reactor_class
import copy

import matplotlib.animation as animation
env = reactor_class(test = True,ns = 120)

ns = 120
Ca_des = [0.87 for i in range(int(2*ns/5))] + [0.91 for i in range(int(ns/5))] + [0.85 for i in range(int(2*ns/5))]                     
T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]
# Get file paths in model log directory 
try:
  filepath = []
  for subdir, dirs, files in os.walk('.\logs\SAC'):
      for file in files:
          #print os.path.join(subdir, file)
          filepath.append(subdir + os.sep + file)
   
except StopIteration:
    print("No files found in the directory")

n_files = len(files)
reps = 2
Ca_eval_PG = np.zeros((ns,reps,n_files))
T_eval_PG = np.zeros((ns,reps,n_files))
Tc_eval_PG = np.zeros((ns,reps,n_files))
ks_eval_PG = np.zeros((7,ns,reps,n_files))
ISE_Ca = np.zeros((n_files))
ISE_T = np.zeros((n_files))
SP = np.array([Ca_des,T_des])

for model_i in range(n_files):
    model = SAC.load(filepath[model_i])
    for r_i in range(reps):
      s_norm,_ = env.reset()
      s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
      Ca_eval_PG[0,r_i,model_i] = s[0]
      T_eval_PG[0,r_i,model_i] = s[1]
      Tc_eval_PG[0,r_i,model_i] = 300.0
      a_policy = model.predict(s_norm,deterministic=True)[0]
      
      a_sim = a_policy
      for ks_i in range(0,3):
          a_sim[ks_i] = (a_sim[ks_i])
          
      for ks_i in range(3,6):
          a_sim[ks_i] = (a_sim[ks_i])
          
      a_sim[6] = (a_sim[ks_i]) + 293
      ks_eval_PG[:,0,r_i,model_i] = a_sim
      r_tot = 0
      for i in range(1,ns):    
        if i % 5 == 0:
          a_policy = model.predict(s_norm,deterministic=True)[0]
          # [-1,1] -> [0,1]
          a_sim = a_policy
          for ks_i in range(0,3):
              a_sim[ks_i] = (a_sim[ks_i])
              
          for ks_i in range(3,6):
              a_sim[ks_i] = (a_sim[ks_i])
            
          a_sim[6] = (a_sim[ks_i]) + 293
        ks_eval_PG[:,i,r_i,model_i] = a_sim
        
        
        a_copy = copy.deepcopy(a_sim)
        s_norm, r, done, info,_ = env.step(a_policy)
        a_sim = a_copy
        r_tot += r
        s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
        Ca_eval_PG[i,r_i,model_i] = s[0]
        T_eval_PG[i,r_i,model_i] = s[1]
        Tc_eval_PG[i,r_i,model_i] = env.u_history[-1]
      ISE_Ca[model_i] = np.sum((Ca_des-Ca_eval_PG[:,0,model_i])**2)
      ISE_T[model_i] = np.sum((T_des-T_eval_PG[:,0,model_i])**2)
      
# Create animation
plt.rcParams['text.usetex'] = 'True'
fig, axs = plt.subplots(6, 1, sharex=True,figsize = (10,12))

t = np.linspace(0, 25, ns)

# Assuming ca, T, and Tc are your data arrays
ca = Ca_eval_PG[:, 0, 0]
T = T_eval_PG[:, 0, 0]
Tc = Tc_eval_PG[:, 0, 0]
ca_kp = ks_eval_PG[0,:,0,0]
ca_ki = ks_eval_PG[1,:,0,0]
ca_kd = ks_eval_PG[2,:,0,0]
T_kp = ks_eval_PG[3,:,0,0]
T_ki = ks_eval_PG[4,:,0,0]
T_kd = ks_eval_PG[5,:,0,0]
kb = ks_eval_PG[6,:,0,0]

# Initialize line plots

line1, = axs[0].plot(t, ca, c='tab:red', label='$C_a$')
axs[0].step(t,Ca_des,linestyle = 'dashed',color = 'black',label = 'SP')
line2, = axs[1].plot(t, T, c='tab:green', label='$T$')
axs[1].step(t,T_des,linestyle = 'dashed',color = 'black',label = 'SP')
line3, = axs[2].step(t, Tc, c='tab:orange', label='$T_c$',linestyle = 'dashed')

line4, = axs[3].step(t, ca_kp, c='tab:red', label='$k_p$')
line5, = axs[3].step(t, ca_ki, c='tab:green', label='$k_i$')
line6, = axs[3].step(t, ca_kd, c='tab:orange', label='$k_d$')

line7, = axs[4].step(t, T_kp, c='tab:red', label='$k_p$')
line8, = axs[4].step(t, T_ki, c='tab:green', label='$k_i$')
line9, = axs[4].step(t, T_kd, c='tab:orange', label='$k_d$')

line10, = axs[5].step(t, kb, c='tab:red', label='$k_b$')


policy_update = axs[0].annotate('', xy=(0.02, 0.85), xycoords='axes fraction')
Ca_ISE_Anno = axs[0].annotate('', xy=(0.02, 0.1), xycoords='axes fraction')
T_ISE_Anno = axs[0].annotate('', xy=(0.15, 0.1), xycoords='axes fraction')
axs[0].set(ylabel='$C_a$ [mol/m$^3$]')
axs[1].set(ylabel='$T$')
axs[3].set(ylabel = '$C_a$ PID Parameters')
axs[4].set(ylabel = '$T$ PID Parameters')
axs[5].set(ylabel = 'Baseline PID Parameters')
axs[2].set(xlabel='Time [$s$]', ylabel='$T_c$')

axs[5].set_ylim(np.min(ks_eval_PG[6,:,0,:]),np.max(ks_eval_PG[6,:,0,:]))
for ax in axs:
  ax.legend(loc = 'right')
  ax.set_xlim(left = 0)


def update(frame):
    # for each frame, update the data stored on each artist
    ca_f = Ca_eval_PG[:, 0, frame]
    T_f = T_eval_PG[:, 0,frame]
    Tc_f =Tc_eval_PG[:, 0, frame]
    ca_kp_f = ks_eval_PG[0,:,0,frame]
    ca_ki_f = ks_eval_PG[1,:,0,frame]
    ca_kd_f = ks_eval_PG[2,:,0,frame]
    T_kp_f = ks_eval_PG[3,:,0,frame]
    T_ki_f = ks_eval_PG[4,:,0,frame]
    T_kd_f = ks_eval_PG[5,:,0,frame]
    kb_f = ks_eval_PG[6,:,0,frame]

    # update the line plots:
    line1.set_data(t, ca_f)
    line2.set_data(t, T_f)
    line3.set_data(t,Tc_f)
    line4.set_data(t, ca_kp_f)
    line5.set_data(t, ca_ki_f)
    line6.set_data(t, ca_kd_f)
    line7.set_data(t, T_kp_f)
    line8.set_data(t, T_ki_f)
    line9.set_data(t, T_kd_f)
    line10.set_data(t, kb_f)
    policy_update.set_text(f'Number of policy updates: {frame*100+100}')
    Ca_ISE_Anno.set_text(f'ISE $C_a$:{round(ISE_Ca[frame],4)}')
    T_ISE_Anno.set_text(f'ISE $T$:{round(ISE_T[frame], 1)}')
    return line1, line2, line3,line4,line5,line6,line7,line8,line9,line10,policy_update,Ca_ISE_Anno,T_ISE_Anno

ani = animation.FuncAnimation(fig, update, frames=n_files, blit=True)

plt.show()
