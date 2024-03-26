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
import matplotlib.gridspec as gridspec
# Initialise Env
ns = 150
env = RSR(ns,test=True,plot=False)
def rollout(ns,policy,reps):    

    states = np.zeros([env.Nx,ns,reps])
    actions = np.zeros([env.action_space.low.shape[0],ns,reps])
    rewards = np.zeros([1,reps])
    controls = np.zeros([env.action_space_unnorm.low.shape[0]+1,ns,reps])
    for r_i in range(reps):
        tot_reward = 0
        s,_ = env.reset()
        
        for i in range(ns):
        
            a = policy.predict(torch.tensor(s),deterministic = True)[0]
       
            
            s, reward, done, _,control = env.step(a)
            states[:,i,r_i] = control['state']
            actions[:,i,r_i] = control['PID_Action']
            tot_reward += reward
            controls[:,i,r_i] = control['control_in']
        rewards[:,r_i] = tot_reward
    print(f'PG-RL:{np.mean(rewards,axis=1)}')   
    return states, actions, tot_reward,controls

class Net(torch.nn.Module):
  def __init__(self, n_fc1, n_fc2, activation,n_layers,deterministic, **kwargs):
    super(Net, self).__init__()

    # Unpack the dictionary
    self.args     = kwargs
    self.dtype    = torch.float
    self.use_cuda = torch.cuda.is_available()
    self.device   = torch.device("cpu")
    self.deterministic = deterministic  
    self.input_size = 9 #State size: Ca, T, Ca setpoint and T setpoint
    self.output_sz  = 12 #Output size: Reactor Ks size
    self.n_layers = torch.nn.ModuleList()
    self.hs1        = n_fc1                                    # !! parameters
    self.hs2        = n_fc2                                      # !! parameter

    # defining layer
    self.hidden1 = torch.nn.Linear(self.input_size, self.hs1,bias=True)
    self.act = activation()
    self.hidden2 = torch.nn.Linear(self.hs1, self.hs2,bias=True)
    for i in range(0,n_layers):
      linear_layer = torch.nn.Linear(self.hs2,self.hs2)
      self.n_layers.append(linear_layer)
    self.output_mu = torch.nn.Linear(self.hs2, self.output_sz, bias=True)
    self.output_std = torch.nn.Linear(self.hs2, self.output_sz, bias=True)

  def predict(self, x):

    x = x.float()
    y           = self.act(self.hidden1(x))
    y           = self.act(self.hidden2(y))        
    mu = self.output_mu(y)
    log_std = self.output_std(y) 
    dist = torch.distributions.Normal(mu, log_std.exp()+ 1e-6)
    out = dist.sample()                                         
    y = F.tanh(out) #[-1,1]
    if self.deterministic:
      y = torch.tanh(mu)

    return y,None
def plot_simulation(s_GS,s_const,s_ea,s_pg,a_GS,a_const,a_ea,a_pg,c_GS,c_const,c_ea,c_pg,ns,ISE_GS,ISE_const):
    #Enable LaTeX for the plots
   

    #Load Data
    GS_a_data = np.array(a_GS)
    GS_c_data = np.array(c_GS)
    
    const_a_data = np.array(a_const)
    const_c_data = np.array(c_const)

    ea_a_data = np.array(a_ea)
    ea_c_data = np.array(c_ea)

    pg_a_data = np.array(a_pg)
    pg_c_data = np.array(c_pg)

    SP = SP_M = env.SP_test[0,:]
    GS_s_data = [np.array(s_GS)[i,:,:]  for i in [0,4,8,1,2,3,5,6,7,9,10,11]]
    const_s_data = [np.array(s_const)[i,:,:]  for i in [0,4,8,1,2,3,5,6,7,9,10,11]]
    ea_s_data = [np.array(s_ea)[i,:,:]  for i in [0,4,8,1,2,3,5,6,7,9,10,11]]
    pg_s_data = [np.array(s_pg)[i,:,:]  for i in [0,4,8,1,2,3,5,6,7,9,10,11]]
    
    #Colours, Titles and Labels
    titles = ['Level', 'Component Fractions', 'Actions', 'Control Inputs']
    control_labels = ['$F_R$', '$F_M$', '$B$', '$D$']
    labels = ['Reactor', 'Storage', 'Flash', '$x_{1,R}$', '$x_{2,R}$', '$x_{3,R}$', '$x_{1,M}$', '$x_{2,M}$', '$x_{3,M}$', '$x_{1,B}$', '$x_{2,B}$', '$x_{3,B}$']
    PID_labels = ['$k_p$', '$k_i$', '$k_d$', '$k_b$']
    colors_PID = ['tab:blue', 'tab:orange', 'tab:green']
    colors_method =['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    linestyles_state = ['-', '--', ':'] 
    linestyles_control = ['-', '--']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange']
    
    t = np.linspace(0, 20, ns)    

    # DFO Holdup Plots
    fig, axs = plt.subplots(1, 3, figsize=(14, 10))

    axs[0].plot(t,np.median(GS_s_data[0],axis = 1), color=colors_method[0],linestyle = linestyles_state[0], label= 'GS '+labels[0])
    axs[0].plot(t,np.median(const_s_data[0],axis = 1), colors_method[1],linestyle = linestyles_state[0], label='Constant ' + labels[0])
    axs[0].plot(t,np.median(pg_s_data[0],axis = 1), colors_method[2],linestyle = linestyles_state[0], label='PG-RL ' + labels[0])
    axs[0].plot(t,np.median(ea_s_data[0],axis = 1), colors_method[3],linestyle = linestyles_state[0], label='EA-RL ' + labels[0])
    axs[0].fill_between(t,np.min(const_s_data[0],axis = 1), np.max(const_s_data[0],axis = 1),color =colors_method[1],alpha = 0.5,edgecolor = 'none')
    axs[0].fill_between(t,np.min(GS_s_data[0],axis = 1),np.max(GS_s_data[0],axis = 1),color =colors_method[0],alpha = 0.5,edgecolor = 'none')
    axs[0].fill_between(t,np.min(pg_s_data[0],axis = 1), np.max(pg_s_data[0],axis = 1),color =colors_method[2],alpha = 0.5,edgecolor = 'none')
    axs[0].fill_between(t,np.min(ea_s_data[0],axis = 1),np.max(ea_s_data[0],axis = 1),color =colors_method[3],alpha = 0.5,edgecolor = 'none')
    axs[0].set_ylabel('Vessel Holdup')
    axs[0].step(t, env.SP_test[0,:], 'k-.', where  = 'post',label='SP')
    axs[0].grid(True)
    axs[0].set_xlim(np.min(t),np.max(t))
    axs[0].set_title('Holdups', y=1.3)

    axs[1].plot(t,np.median(GS_s_data[1],axis = 1), color=colors_method[0],linestyle = linestyles_state[1], label= 'GS '+labels[1])
    axs[1].plot(t,np.median(const_s_data[1],axis = 1), colors_method[1],linestyle = linestyles_state[1], label='Constant ' + labels[1])
    axs[1].plot(t,np.median(pg_s_data[1],axis = 1), colors_method[2],linestyle = linestyles_state[1], label='PG-RL ' + labels[1])
    axs[1].plot(t,np.median(ea_s_data[1],axis = 1), colors_method[3],linestyle = linestyles_state[1], label='EA-RL ' + labels[1])
    axs[1].fill_between(t,np.min(const_s_data[1],axis = 1), np.max(const_s_data[1],axis = 1),color =colors_method[1],alpha = 0.5,edgecolor = 'none')
    axs[1].fill_between(t,np.min(GS_s_data[1],axis = 1),np.max(GS_s_data[1],axis = 1),color =colors_method[0],alpha = 0.5,edgecolor = 'none')
    axs[1].fill_between(t,np.min(pg_s_data[1],axis = 1), np.max(pg_s_data[1],axis = 1),color =colors_method[2],alpha = 0.5,edgecolor = 'none')
    axs[1].fill_between(t,np.min(ea_s_data[1],axis = 1),np.max(ea_s_data[1],axis = 1),color =colors_method[3],alpha = 0.5,edgecolor = 'none')
    axs[1].set_ylabel('Vessel Holdup')
    axs[1].step(t, env.SP_test[0,:], 'k-.', where  = 'post',label='SP')
    axs[1].grid(True)
    axs[1].set_xlim(np.min(t),np.max(t))
    axs[1].set_title('Holdups', y=1.3)

    axs[2].plot(t,np.median(GS_s_data[2],axis = 1), color=colors_method[0],linestyle = linestyles_state[2], label= 'GS '+labels[2])
    axs[2].plot(t,np.median(const_s_data[2],axis = 1), colors_method[1],linestyle = linestyles_state[2], label='Constant ' + labels[2])
    axs[2].plot(t,np.median(pg_s_data[2],axis = 1), colors_method[2],linestyle = linestyles_state[2], label='PG-RL ' + labels[2])
    axs[2].plot(t,np.median(ea_s_data[2],axis = 1), colors_method[3],linestyle = linestyles_state[2], label='EA-RL ' + labels[2])
    axs[2].fill_between(t,np.min(const_s_data[2],axis = 1), np.max(const_s_data[2],axis = 1),color =colors_method[0],alpha = 0.5,edgecolor = 'none')
    axs[2].fill_between(t,np.min(GS_s_data[2],axis = 1),np.max(GS_s_data[2],axis = 1),color =colors_method[1],alpha = 0.5,edgecolor = 'none')
    axs[2].fill_between(t,np.min(pg_s_data[2],axis = 1), np.max(pg_s_data[2],axis = 1),color =colors_method[2],alpha = 0.5,edgecolor = 'none')
    axs[2].fill_between(t,np.min(ea_s_data[2],axis = 1),np.max(ea_s_data[2],axis = 1),color =colors_method[3],alpha = 0.5,edgecolor = 'none')
    axs[2].set_ylabel('Vessel Holdup')
    axs[2].step(t, env.SP_test[0,:], 'k-.', where  = 'post',label='SP')
    axs[2].grid(True)
    axs[2].set_xlim(np.min(t),np.max(t))
    axs[2].set_title('Holdups', y=1.3)
    axs[2].legend(loc='upper center',
                  ncol=3, frameon=False)
    plt.show()
    fig, axs = plt.subplots(1, 2, figsize=(14, 10))
    for i in range(2):
       axs[0].step(t, np.median(GS_c_data[i,:,:],axis =1), where='post', linestyle=linestyles_control[i], color=colors_method[0], label='GS ' +control_labels[i])
       axs[0].fill_between(t,np.min(GS_c_data[i,:,:],axis = 1),np.max(GS_c_data[i,:,:],axis = 1),color = colors_method[0],alpha = 0.2,edgecolor = 'none')
       axs[0].step(t, np.median(const_c_data[i,:,:],axis =1), where='post', linestyle=linestyles_control[i],color=colors_method[1], label='const ' +control_labels[i])
       axs[0].fill_between(t,np.min(const_c_data[i,:,:],axis = 1),np.max    (const_c_data[i,:,:],axis = 1),color = colors_method[1],alpha = 0.2,edgecolor = 'none')
       axs[0].step(t, np.median(pg_c_data[i,:,:],axis =1), where='post', linestyle=linestyles_control[i], color=colors_method[2], label='PG-RL ' +control_labels[i])
       axs[0].fill_between(t,np.min(pg_c_data[i,:,:],axis = 1),np.max(pg_c_data[i,:,:],axis = 1),color = colors_method[2],alpha = 0.2,edgecolor = 'none')
       axs[0].step(t, np.median(ea_c_data[i,:,:],axis =1), where='post',linestyle=linestyles_control[i], color=colors_method[3], label='EA-RL ' +control_labels[i])
       axs[0].fill_between(t,np.min(ea_c_data[i,:,:],axis = 1),np.max    (ea_c_data[i,:,:],axis = 1),color = colors_method[3],alpha = 0.2,edgecolor = 'none')
    axs[0].set_ylabel('Flowrate (h$^{-1}$)')
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=4,frameon=False)
    axs[0].grid(True)
    axs[0].set_xlim(np.min(t),np.max(t))

    for i in range(2,4):
        axs[1].step(t, np.median(GS_c_data[i, :, :], axis =1),  where='post',  linestyle=linestyles_control[i-2],  color=colors_method[0],  label='GS ' +control_labels[i])
        axs[1].fill_between(t, np.min(GS_c_data[i, :, :], axis = 1), np.max(GS_c_data[i, :, :], axis = 1), color = colors_method[0], alpha = 0.2, edgecolor = 'none')
        axs[1].step(t,  np.median(const_c_data[i, :, :], axis =1),  where='post', linestyle=linestyles_control[i-2], color=colors_method[1],  label='const ' +control_labels[i])
        axs[1].fill_between(t,  np.min(const_c_data[i, :, :], axis = 1), np.max(const_c_data[i, :, :], axis = 1), color = colors_method[1], alpha = 0.2, edgecolor = 'none')
        axs[1].step(t, np.median(pg_c_data[i, :, :], axis =1),  where='post',  linestyle=linestyles_control[i-2],  color=colors_method[2],  label='PG-RL ' +control_labels[i])
        axs[1].fill_between(t, np.min(pg_c_data[i, :, :], axis = 1), np.max(pg_c_data[i, :, :], axis = 1), color = colors_method[2], alpha = 0.2, edgecolor = 'none')
        axs[1].step(t,  np.median(ea_c_data[i, :, :], axis =1),  where='post', linestyle=linestyles_control[i-2], color=colors_method[3],  label='EA-RL ' +control_labels[i])
        axs[1].fill_between(t,  np.min(ea_c_data[i, :, :], axis = 1), np.max(ea_c_data[i, :, :], axis = 1), color = colors_method[3], alpha = 0.2, edgecolor = 'none')
    axs[1].set_ylabel('Flowrate (h$^{-1}$)')
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                    ncol=4, frameon=False)
    axs[1].grid(True)
    axs[1].set_xlim(np.min(t),np.max(t))
    plt.show()
    
    
    fig, axs = plt.subplots(1, 4, figsize=(14, 10))

    # PID Action Plots
    for i in range(0,3):
        axs[0].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i], color=colors_method[0], label='GS ' + PID_labels[i % len(PID_labels)])
        axs[0].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_method[1],linestyle=linestyles_state[i], label='Constant ' + PID_labels[i % len(PID_labels)])
        axs[0].step(t, np.median(pg_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i], color=colors_method[2], label='PG-RL ' + PID_labels[i % len(PID_labels)])
        axs[0].step(t, np.median(ea_a_data[i,:,:],axis =1), where='post', color=colors_method[3], linestyle = linestyles_state[i],label='EA-RL ' + PID_labels[i % len(PID_labels)])
    axs[0].set_ylabel('$F_R$ PID Action')
    axs[0].legend(loc='upper center',
          ncol=4,frameon=False)
    axs[0].grid(True)
    axs[0].set_xlim(np.min(t),np.max(t))
    for i in range(3,6):
        axs[1].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i-3], color=colors_method[1], label='GS ' +PID_labels[i % len(PID_labels)])
        axs[1].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_method[0],linestyle = linestyles_state[i-3] ,label='const ' +PID_labels[i % len(PID_labels)])
        axs[1].step(t, np.median(pg_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i-3], color=colors_method[2], label='PG-RL ' +PID_labels[i % len(PID_labels)])
        axs[1].step(t, np.median(ea_a_data[i,:,:],axis =1), where='post', color=colors_method[3], linestyle = linestyles_state[i-3],label='EA-RL ' +PID_labels[i % len(PID_labels)])
    axs[1].set_ylabel('$F_M$ PID Action')
    axs[1].grid(True)
    axs[1].set_xlim(np.min(t),np.max(t))
    #ax5.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)
    for i in range(6,9):
        axs[2].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i-6], color=colors_method[1], label='GS' +PID_labels[i % len(PID_labels)])
        axs[2].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_method[0],linestyle=linestyles_state[i-6], label='const' +PID_labels[i % len(PID_labels)])
        axs[2].step(t, np.median(pg_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i-6], color=colors_method[2], label='PG-RL ' +PID_labels[i % len(PID_labels)])
        axs[2].step(t, np.median(ea_a_data[i,:,:],axis =1), where='post', color=colors_method[3],linestyle=linestyles_state[i-6], label='EA-RL ' +PID_labels[i % len(PID_labels)])
    axs[2].set_ylabel('$B$ PID Action')
    axs[2].grid(True)
    axs[2].set_xlim(np.min(t),np.max(t))
    #ax6.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)
    
    for i in range(9,12):
        axs[3].step(t, np.median(GS_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i-9], color=colors_method[1], label='GS' +PID_labels[i % len(PID_labels)])
        axs[3].step(t, np.median(const_a_data[i,:,:],axis =1), where='post', color=colors_method[0], linestyle=linestyles_state[i-9],label='const' +PID_labels[i % len(PID_labels)])
        axs[3].step(t, np.median(pg_a_data[i,:,:],axis =1), where='post', linestyle=linestyles_state[i-9], color=colors_method[2], label='PG-RL ' +PID_labels[i % len(PID_labels)])
        axs[3].step(t, np.median(ea_a_data[i,:,:],axis =1), where='post', color=colors_method[3], linestyle=linestyles_state[i-9],label='EA-RL ' +PID_labels[i % len(PID_labels)])
    axs[3].set_ylabel('$D$ PID Action')
    axs[3].grid(True)
    axs[3].set_xlim(np.min(t),np.max(t))
    #axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)

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
              if i % 50 == 0:
                  Ks_i += 1
            a = Ks[Ks_i*12:(Ks_i+1)*12]
            s, reward, done, _,control = env.step(a)
            states[:,i,r_i] = control['state']
            actions[:,i,r_i] = control['PID_Action']
            tot_reward += reward
            controls[:,i,r_i] = control['control_in']
        rewards[:,r_i] = tot_reward
    tot_rew = np.mean(rewards, axis = 1)
    
    return states, actions, tot_rew,controls
def rollout_DFO(ns,policy,reps,test):   
    done = False
    states = np.zeros([env.Nx,ns,reps])
    actions = np.zeros([env.action_space.low.shape[0],ns,reps])
    rewards = np.zeros([1,reps])
    controls = np.zeros([env.action_space_unnorm.low.shape[0],ns,reps])
    tot_reward = 0
    for r_i in range(reps):
        tot_reward = 0
        s,_ = env.reset()
        states[:,0,r_i] = env.state[:env.Nx]
        a_0 = (policy.predict(torch.tensor(s))[0].detach().numpy()/2+1)/2
        a_0 =  a_0* (env.PID_space.high - env.PID_space.low) + env.PID_space.low
        actions[:,0,r_i] = a_0 
        control_0 = (env.action_space_unnorm.high - env.action_space_unnorm.low)/2
        controls[:,0,r_i] = control_0[:4] 
        i = 0
        if test:
            n_s = 150
        else:
            n_s= 450
        for i in range(n_s-1):
            a = policy.predict(torch.tensor(s))[0].detach().numpy()
            s, reward, done, _,control = env.step(a)
            tot_reward += reward
            if done:
                break
          
            states[:,i+1,r_i] = control['state']
            actions[:,i+1,r_i] = control['PID_Action']
            controls[:,i+1,r_i] = control['control_in'][:4]
            
            i += 1
        rewards[:,r_i] = tot_reward 
    if test:
        print(f'DFO-RL: {np.mean(rewards,axis =1)}')
    return states, actions, tot_reward,controls
# Rollout const
Ks_const = np.load('GS_const.npy')
s_const,a_const,r_const,c_const = rollout_test(Ks_const, ns, PID = 'const')

# Rollout GS
Ks_GS = np.load('GS.npy')
s_GS,a_GS,r_GS,c_GS = rollout_test(Ks_GS, ns, PID = 'GS')

# Rollout DFO-RL
best_policy_sd = torch.load('DFO_SP_0703')
policy_plot = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1,deterministic = True) # Deterministic for plotting
policy_plot.load_state_dict(best_policy_sd)
s_ea,a_ea,r_ea,c_ea = rollout_DFO(ns,policy_plot,10,True)

# Rollout PG-RL
model = SAC.load('SAC_1903.zip')
s_pg, a_pg, r_pg, c_pg = rollout(ns, model, 10)

# Plot Comparison
print('const ISE:',np.round(r_const,2)*-1)

print('GS ISE:',np.round(r_GS,2)*-1)
plot_simulation(s_GS,s_const,s_ea,s_pg,a_GS,a_const,a_ea,a_pg,c_GS,c_const,c_ea,c_pg,ns,r_GS[0]*-1,r_const[0]*-1)
