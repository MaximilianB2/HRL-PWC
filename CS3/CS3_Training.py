#Import Libraries
import gymnasium as gym
from gymnasium import spaces 
from stable_baselines3 import SAC, PPO
from casadi import *
import numpy as np
import torch
from stable_baselines3 import SAC, PPO
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_pso import ParticleSwarmOptimizer
import copy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from typing import Callable
from RSR_Model_1602 import RSR
# Create Environment
ns = 300
env = RSR(ns,test=False,plot=False)


def sample_uniform_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min \
              for k, v in params_prev.items()}
    return params

def sample_local_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min + v \
              for k, v in params_prev.items()}
    return params

  
def plot_simulation(states,actions, control_inputs,ns):
    plt.rcParams['text.usetex'] = 'True'
    
    actions = np.array(actions)
    control_inputs = np.array(control_inputs)
    SP = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]]).reshape(ns,1)
    SP_M = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]]).reshape(ns,1)
    data = [np.array(states)[i,:,:] for i in [0,4,8,1,2,3,5,6,7,9,10,11]]
   
    titles = ['Level', 'Component Fractions', 'Actions', 'Control Inputs']
    control_labels = ['$F_R$', '$F_M$', '$B$', '$D$']
    labels = ['Reactor', 'Storage', 'Flash', '$x_{1,R}$', '$x_{2,R}$', '$x_{3,R}$', '$x_{1,M}$', '$x_{2,M}$', '$x_{3,M}$', '$x_{1,B}$', '$x_{2,B}$', '$x_{3,B}$']
    PID_labels = ['$k_p$', '$k_i$', '$k_d$', '$k_b$']
    colors_PID = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange']

    t = np.linspace(0, 20, ns)    
    fig, axs = plt.subplots(4, 2, figsize=(10, 12))
    for i in range(8):  # Loop over all 7 axes
        ax = axs[i//2, i%2]  # Get current axes
        ax.grid(True)  # Add grid
        ax.set_xlim(left=0, right=20)  # Set x-axis limits
        ax.set_xlabel('Time (h)')
    #Level Plots
    for i in range(3):
        axs[0,0].plot(t,np.median(data[i],axis= 1), color=colors[i], label=labels[i])
        axs[0,0].fill_between(t,np.min(data[i],axis = 1),np.max(data[i],axis = 1),color = colors[i],alpha = 0.2,edgecolor = 'none')
        axs[0,0].set_ylabel('Vessel Holdup')
    #axs[0,0].step(t, SP, 'k--', where  = 'post',label='SP$_R$ \& SP$_B$')
    axs[0,0].step(t, SP_M, 'k-.', where  = 'post',label='SP$_M$')
   
    axs[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3,frameon=False)
    #axs[0,0].set_ylim(20.5, 21.5)
    
    #Component Fraction Plots
    for i in range(3, len(data)):
        axs[0,1].plot(t,np.median(data[i],axis= 1), color=colors[i], label=labels[i])
        axs[0,1].fill_between(t,np.min(data[i],axis = 1),np.max(data[i],axis = 1),color = colors[i],alpha = 0.2,edgecolor = 'none')
    axs[0,1].set_ylabel('Liquid Component Fraction')
    axs[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
          ncol=3,frameon=False)
    #PID Action Plots
    for i in range(0,4):
        axs[1,0].step(t, np.median(actions[i,:,:],axis = 1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
        axs[1,0].fill_between(t,np.min(actions[i,:,:],axis = 1),np.max(actions[i,:,:],axis = 1),color = colors_PID[i % len(colors_PID)],alpha = 0.2,edgecolor = 'none')
    axs[1,0].set_ylabel('$F_R$ PID Action')
    axs[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)
    for i in range(4,8):
        axs[1,1].step(t, np.median(actions[i,:,:],axis = 1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
        axs[1,1].fill_between(t,np.min(actions[i,:,:],axis = 1),np.max(actions[i,:,:],axis = 1),color = colors_PID[i % len(colors_PID)],alpha = 0.2,edgecolor = 'none')
    axs[1,1].set_ylabel('$F_M$ PID Action')
    axs[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)
    for i in range(8,12):
        axs[2,0].step(t, np.median(actions[i,:,:],axis = 1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
        axs[2,0].fill_between(t,np.min(actions[i,:,:],axis = 1),np.max(actions[i,:,:],axis = 1),color = colors_PID[i % len(colors_PID)],alpha = 0.2,edgecolor = 'none')
    axs[2,0].set_ylabel('$B$ PID Action')
    axs[2,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)
    
    for i in range(12,16):
        axs[2,1].step(t, np.median(actions[i,:,:],axis = 1), where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
        axs[2,1].fill_between(t,np.min(actions[i,:,:],axis = 1),np.max(actions[i,:,:],axis = 1),color = colors_PID[i % len(colors_PID)],alpha = 0.2,edgecolor = 'none')
    axs[2,1].set_ylabel('$D$ PID Action')
    axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)

    for i in range(2):
        axs[3,0].step(t, np.median(control_inputs[i,:,:],axis = 1), where='post', linestyle='dashed', color=colors[i % len(colors)], label=control_labels[i])
        axs[3,0].fill_between(t,np.min(control_inputs[i,:,:],axis = 1),np.max(control_inputs[i,:,:],axis = 1),color = colors[i % len(colors)],alpha = 0.2,edgecolor = 'none')
    axs[3,0].set_ylabel('Flowrate (h$^{-1}$)')
    axs[3,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=10,frameon=False)
   
 
    for i in range(2,4):
        axs[3,1].step(t, np.median(control_inputs[i,:,:],axis = 1), where='post', linestyle='dashed', color=colors[i % len(colors)], label=control_labels[i])
        axs[3,1].fill_between(t,np.min(control_inputs[i,:,:],axis = 1),np.max(control_inputs[i,:,:],axis = 1),color = colors[i % len(colors)],alpha = 0.2,edgecolor = 'none')
    axs[3,1].set_ylabel('Flowrate (h$^{-1}$)')
    axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=10,frameon=False)
    
    plt.subplots_adjust(hspace=0.5)
   
    plt.show()

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
        print(tot_reward)
        rewards[:,r_i] = tot_reward
        
    return states, actions, tot_reward,controls


def rollout_test(ns,Ks):
    env = RSR(ns,test=False,plot=False)
    s,_ = env.reset()
    done = False
    states = []
    actions = []
    rewards = []
    controls = []
    tot_rew = 0
    
    while not done:
        action = Ks
        state, reward, done, _,control = env.step(action)
        #un normalise state
        state = state * (env.observation_space.high - env.observation_space.low) + env.observation_space.low
        states.append(control['state'])
        actions.append(control['PID_Action'])
        rewards.append(reward)
        controls.append(control['control_in'])
        tot_rew += reward
    return states, actions, tot_rew,controls


def rollout_DFO(ns,policy,reps,test):   
    done = False
    states = np.zeros([env.Nx,ns,reps])
    actions = np.zeros([env.action_space.low.shape[0],ns,reps])
    rewards = np.zeros([1,reps])
    controls = np.zeros([env.action_space_unnorm.low.shape[0]+1,ns,reps])
    tot_reward = 0
    for r_i in range(reps):
        tot_reward = 0
        s,_ = env.reset()
        i = 0
        if test:
            n_s = 300
        else:
            n_s= 900
        for i in range(n_s):
            a = policy.predict(torch.tensor(s))[0].detach().numpy()
            s, reward, done, _,control = env.step(a)
            if test:
                states[:,i,r_i] = control['state']
                actions[:,i,r_i] = control['PID_Action']
                controls[:,i,r_i] = control['control_in']
            tot_reward += reward
            i += 1
        rewards[:,r_i] = tot_reward 
        if test:
            print(tot_reward)
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
    self.input_size = 10 #State size: Ca, T, Ca setpoint and T setpoint
    self.output_sz  = 16 #Output size: Reactor Ks size
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
  
def criterion(policy,ns):
  s,a,r,c = rollout_DFO(ns,policy,1,False)
  r = np.array(r)
  r_tot = -1*np.sum(r)
  
  global r_list
  global p_list
  global r_list_i
  r_list.append(r_tot)
  r_list_i.append(r_tot)
  p_list.append(policy)
  return r_tot

# policy = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1,deterministic = False)


# # #Training Loop Parameters
# k0     = 7.2e10 # Pre-exponential factor (1/sec)
# UA     = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
# old_swarm  = 1e8
# new_swarm = 0
# tol = 0.01

# env = RSR(ns,test = False,plot= False)

# max_iter = 30

# policy_list = np.zeros(max_iter)
# reward_list = np.zeros(max_iter)
# old_swarm = np.zeros(max_iter)
# best_reward = 1e8
# i = 0
# r_list = []
# r_list_i =[]
# p_list  =[]

# evals_rs = 15

# params = policy.state_dict()
# #Random Search
# print('Random search to find good initial policy...')
# max_param = 0.3
# min_param = -1*max_param
# for policy_i in range(evals_rs):
#     # sample a random policy
#     NNparams_RS  = sample_uniform_params(params,max_param, min_param)
#     # consruct policy to be evaluated
#     policy.load_state_dict(NNparams_RS)
#     # evaluate policy
#     r = criterion(policy,ns)
#     #Store rewards and policies
#     if r < best_reward:
#         best_policy = p_list[r_list.index(r)]
#         best_reward = r
#         init_params= copy.deepcopy(NNparams_RS)
# policy.load_state_dict(init_params)
# #PSO Optimisation paramters
# optim = ParticleSwarmOptimizer(policy.parameters(),
#                                inertial_weight=0.5,
#                                num_particles=5,
#                                max_param_value=max_param,
#                                min_param_value=min_param)
# print('Best reward after random search:', best_reward)
# print('PSO Algorithm...')
# while i < max_iter and abs(best_reward - old_swarm[i]) > tol :
#     if i > 0:
#       old_swarm[i] = min(r_list_i)
#       del r_list_i[:]
#     def closure():
#         # Clear any grads from before the optimization step, since we will be changing the parameters
#         optim.zero_grad()
#         return criterion(policy,ns)
#     optim.step(closure)
#     new_swarm = min(r_list_i)

#     if new_swarm < best_reward:
#       best_reward = new_swarm
#       best_policy = p_list[r_list.index(new_swarm)]
      
#       print('New best reward:', best_reward,'iteration:',i+1,'/',max_iter)
#     i += 1
# print('Finished optimisation')
# print('Best reward:', best_reward)
# print('Saving Best Policy...')
# torch.save(best_policy.state_dict(), 'DFO_best_policy_wNoise_wF0_1602.pth')
# best_policy_sd = best_policy.state_dict()
# # best_policy_sd = torch.load('DFO_best_policy_wNoise_wF0_1602.pth')
# policy_plot = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1,deterministic = True) # Deterministic for plotting
# policy_plot.load_state_dict(best_policy_sd)
# print('Plotting Best Policy...')

# env = RSR(ns,test=True,plot=False)
# s,a,r,c = rollout_DFO(ns,policy_plot,10,True)


# plot_simulation(s,a,c,ns)

# Ks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# s,a,r,c = rollout_test(ns,Ks)

# plot_simulation(s,a,c,ns)
# #

checkpoint_callback = CheckpointCallback(save_freq=1200, save_path="./logs/wNoise_F0",
                                         name_prefix="SAC_model_1602")

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining < 0.7:
          return progress_remaining * initial_value
        elif progress_remaining < 0.3:
          return 0.02
        else:
          return initial_value
    return func

model = SAC('MlpPolicy', env, verbose = 1,learning_rate=linear_schedule(2e-2),device='cuda',seed =int(0))
model.learn(total_timesteps=1e4,callback=checkpoint_callback)

# model = SAC.load('./logs/wNoise/SAC_model_1602_13200_steps.zip')


s, a,r,c = rollout(ns, model,3)


plot_simulation(s,a,c,ns)