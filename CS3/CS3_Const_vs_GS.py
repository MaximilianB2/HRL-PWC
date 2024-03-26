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

# Initialise Env
ns = 150
env = RSR(ns,test=True,plot=False)

def plot_simulation(s_GS,s_const,a_GS,a_const,c_GS,c_const,ns,ISE_GS,ISE_const):
    #Enable LaTeX for the plots
   

    #Load Data
    GS_a_data = np.array(a_GS)
    GS_c_data = np.array(c_GS)
    const_a_data = np.array(a_const)
    const_c_data = np.array(c_const)
    SP = SP_M = env.SP_test[0,:]
    GS_s_data = [np.array(s_GS)[i,:,:]  for i in [0,4,8,1,2,3,5,6,7,9,10,11,12]]
    const_s_data = [np.array(s_const)[i,:,:]  for i in [0,4,8,1,2,3,5,6,7,9,10,11,12]]

    #Colours, Titles and Labels
    titles = ['Level', 'Component Fractions', 'Actions', 'Control Inputs']
    control_labels = ['$F_R$', '$F_M$', '$B$', '$D$','$T_C$']
    labels = ['Reactor', 'Storage', 'Flash', '$x_{1,R}$', '$x_{2,R}$', '$x_{3,R}$', '$x_{1,M}$', '$x_{2,M}$', '$x_{3,M}$', '$x_{1,B}$', '$x_{2,B}$', '$x_{3,B}$']
    PID_labels = ['$k_p$', '$k_i$', '$k_d$', '$k_b$']
    colors_PID = ['tab:blue', 'tab:orange', 'tab:green']
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange']

    t = np.linspace(0, 20, ns)    
    fig, axs = plt.subplots(2, 5, figsize=(14, 10))
    for ax_row in axs:  # Loop over rows of axes
        for ax in ax_row:  # Loop over individual axes in each row
            ax.grid(True)  # Add grid
            ax.set_xlim(left=0, right=20)  # Set x-axis limits
            ax.set_xlabel('Time (h)')

    # DFO Holdup Plots
    
    for i in range(3):
        axs[0,0].plot(t,np.median(GS_s_data[i],axis = 1), color=colors[i], label= labels[i])
        axs[0,0].fill_between(t,np.min(GS_s_data[i],axis = 1),np.max(GS_s_data[i],axis = 1),color = colors[i],alpha = 0.5,edgecolor = 'none')
        axs[0,0].set_ylabel('Vessel Holdup')
        axs[0,0].step(t, env.SP_test[i,:], 'k-.', where  = 'post',label='SP')
    #axs[0,0].step(t, SP, 'k--', where  = 'post',label='SP$_R$ \& SP$_B$')
    axs[0,0].set_title('GS Holdups', y=1.1)
    
    # axs[0,0].axvline(x=t[int((ns-1)/3)], alpha=0.7, linestyle = 'dashed',label = '$F_0$ Disturbance') 
    axs[0, 0].text(0.75, 0.95, 'GS ISE: {:.2f}'.format(ISE_GS), transform=axs[0,0].transAxes, ha='center', va='top', bbox=dict(facecolor='white', edgecolor='white'))
    axs[0,1].text(0.75, 0.95, 'Const ISE: {:.2f}'.format(ISE_const), transform=axs[0,1].transAxes, ha='center', va='top', bbox=dict(facecolor='white', edgecolor='white'))



    # axins = zoomed_inset_axes(axs[0,0], zoom=3, loc='upper right')  
    # for i in range(3):
    #     axins.plot(t,np.median(GS_s_data[i],axis = 1), color=colors[i], label= labels[i])
    #     axins.fill_between(t,np.min(GS_s_data[i],axis = 1),np.max(GS_s_data[i],axis = 1),color = colors[i],alpha = 0.5,edgecolor = 'none')
    # axins.grid(True)
    # axins.axvline(x = t[int((ns-1)/3)], alpha = 0.7, linestyle = 'dashed') 
    # axins.step(t, SP_M, 'k-.', where  = 'post')
    # axins.set_xlim(6, 10)
    # axins.set_ylim(20.75, 21.3)
    # axins.xaxis.set_visible(False)
    # axins.yaxis.set_visible(False)
    # mark_inset(axs[0,0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axs[0,0].legend(loc='upper center', bbox_to_anchor=(1.2, 1.1),
                    ncol=5, frameon=False)
    
    
    #SAC Holdup Plots
    for i in range(3):
        axs[0,1].plot(t,np.median(const_s_data[i],axis = 1), color=colors[i], label='Const ' + labels[i])
        axs[0,1].fill_between(t,np.min(const_s_data[i],axis = 1), np.max(const_s_data[i],axis = 1),color = colors[i],alpha = 0.5,edgecolor = 'none')
        axs[0,1].set_ylabel('Vessel Holdup')
        axs[0,1].step(t, env.SP_test[i,:], 'k-.', where  = 'post',label='SP')
    axs[0,1].set_title('Const Holdups',y = 1.1)
    # axs[0,0].step(t, SP, 'k--', where  = 'post',label='SP$_R$ \& SP$_B$')
    axs[0,1].step(t, SP_M, 'k-.', where  = 'post',label='SP')
    # axs[0,1].axvline(x = t[int((ns-1)/3)], alpha = 0.7, linestyle = 'dashed',label = '$F_0$ Disturbance') 
    # axins = zoomed_inset_axes(axs[0,1], zoom=3, loc='upper right')  
    # for i in range(3):
    #     axins.plot(t,np.median    GS_s_data[i],axis = 1), color=colors[i], label='SAC ' + labels[i])
    #     axins.fill_between(t,np.min   GS_s_data[i],axis = 1),np.max GS_s_data[i],axis = 1),color = colors[i],alpha = 0.2,edgecolor = 'none')
    # axins.grid(True)
    # axins.axvline(x = t[int((ns-1)/3)], alpha = 0.7, linestyle = 'dashed',label = '$F_0$ Disturbance') 
    # axins.step(t, SP_M, 'k-.', where  = 'post')
    # axins.set_xlim(6, 10)
    # axins.set_ylim(20.75, 21.3)
    # axins.xaxis.set_visible(False)
    # axins.yaxis.set_visible(False)
    # mark_inset(axs[0,1], axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # axs[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
    #       ncol=3,frameon=False)
    
    
    # PID Action Plots
    for i in range(0,3):
        axs[1,0].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='GS ' + PID_labels[i % len(PID_labels)])
        axs[1,0].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_PID[i % len(colors_PID)], label='Constant ' + PID_labels[i % len(PID_labels)])
    axs[1,0].set_ylabel('$F_R$ PID Action')
    axs[1,0].legend(loc='upper center', bbox_to_anchor=(2.5, 1.15),
          ncol=3,frameon=False)
    for i in range(3,6):
        axs[1,1].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='GS ' +PID_labels[i % len(PID_labels)])
        axs[1,1].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_PID[i % len(colors_PID)], label='const ' +PID_labels[i % len(PID_labels)])
    axs[1,1].set_ylabel('$F_M$ PID Action')
    #axs[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)
    for i in range(6,9):
        axs[1,2].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='GS' +PID_labels[i % len(PID_labels)])
        axs[1,2].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_PID[i % len(colors_PID)], label='const' +PID_labels[i % len(PID_labels)])
    axs[1,2].set_ylabel('$B$ PID Action')
    #axs[1,2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)
    
    for i in range(12,15):
        axs[1, 3].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='GS' +PID_labels[i % len(PID_labels)])
        axs[1,3].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_PID[i % len(colors_PID)], label='const' +PID_labels[i % len(PID_labels)])
    axs[1,3].set_ylabel('$D$ PID Action')
    #axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)

    for i in range(2):
        axs[0,3].step(t, np.median(GS_c_data[i,:,:],axis =1), where='post', linestyle='dashed', color=colors[i+2 % len(colors)], label='GS ' +control_labels[i])
        axs[0,3].fill_between(t,np.min(GS_c_data[i,:,:],axis = 1),np.max(GS_c_data[i,:,:],axis = 1),color = colors[i+2 % len(colors)],alpha = 0.2,edgecolor = 'none')
        axs[0,3].step(t, np.median(const_c_data[i,:,:],axis =1), where='post', color=colors[i % len(colors)], label='const ' +control_labels[i])
        axs[0,3].fill_between(t,np.min(const_c_data[i,:,:],axis = 1),np.max    (const_c_data[i,:,:],axis = 1),color = colors[i % len(colors)],alpha = 0.2,edgecolor = 'none')
    axs[0,3].set_ylabel('Flowrate (h$^{-1}$)')
    axs[0,3].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=2,frameon=False)

    for i in range(3,5):
        axs[0,2].step(t, np.median(GS_c_data[i, :, :], axis =1),  where='post',  linestyle='dashed',  color=colors[i+2 % len(colors)],  label='GS ' +control_labels[i])
        axs[0, 2].fill_between(t, np.min(GS_c_data[i, :, :], axis = 1), np.max(GS_c_data[i, :, :], axis = 1), color = colors[i+2 % len(colors)], alpha = 0.2, edgecolor = 'none')
        axs[0, 2].step(t,  np.median(const_c_data[i, :, :], axis =1),  where='post',  color=colors[i % len(colors)],  label='const ' +control_labels[i])
        axs[0, 2].fill_between(t,  np.min(const_c_data[i, :, :], axis = 1), np.max(const_c_data[i, :, :], axis = 1), color = colors[i % len(colors)], alpha = 0.2, edgecolor = 'none')
    axs[0, 2].set_ylabel('Flowrate (h$^{-1}$)')
    axs[0,2].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                    ncol=2, frameon=False)
    axs[0,4].plot(t,np.median(const_s_data[10],axis = 1), color = 'tab:blue')
    axs[0,4].plot(t,np.median(GS_s_data[10],axis = 1), color = 'tab:orange')
    axs[0,4].step(t, env.SP_test[3,:], 'k-.', where  = 'post',label='SP')
    axs[1,4].plot(t,np.median(const_s_data[12],axis = 1), color = 'tab:blue')
    axs[1,4].plot(t,np.median(GS_s_data[12],axis = 1), color = 'tab:orange')
    plt.subplots_adjust(hspace = 0.3)
    plt.subplots_adjust(wspace = 0.3)
    plt.savefig('PG_GS_Comparison_wnoise_wF0_1602.pdf')
    plt.show()


def rollout_test(Ks, ns, PID):
    reps = 5
    env = RSR(ns,test=True,plot=False)
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
        for i in range(ns):
            if PID == 'const':
                Ks_i = 0
            if PID == 'GS':
              if i % 25 == 0:
                  Ks_i += 1
            a = np.array([0,0,0,0,0,0,0,0,0,0,0,0,Ks[0+Ks_i*3],Ks[1 + Ks_i*3],Ks[2 + Ks_i*3]])
            #a = Ks[Ks_i*15:(Ks_i+1)*15]
            s, reward, done, _,control = env.step(a)
            states[:,i,r_i] = control['state']
            actions[:,i,r_i] = control['PID_Action']
            tot_reward += reward
            controls[:,i,r_i] = control['control_in']
        rewards[:,r_i] = tot_reward
    tot_rew = np.mean(rewards, axis = 1)
    return states, actions, tot_rew,controls

# Rollout const
Ks_const = np.load('GS_const.npy')
s_const,a_const,r_const,c_const = rollout_test(Ks_const, ns, PID = 'const')

# Rollout GS
Ks_GS = np.load('GS.npy')
s_GS,a_GS,r_GS,c_GS = rollout_test(Ks_GS, ns, PID = 'GS')

# Plot Comparison
print('const ISE:',np.round(r_const,2)*-1)

print('GS ISE:',np.round(r_GS,2)*-1)
plot_simulation(s_GS,s_const,a_GS,a_const,c_GS,c_const,ns,r_GS[0]*-1,r_const[0]*-1)
