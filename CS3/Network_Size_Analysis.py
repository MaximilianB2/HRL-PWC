# Analysis of the agent nn size on the performance
# Import Libraries
from stable_baselines3 import SAC
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Callable
from RSR_Model_1602 import RSR
from datetime import datetime
from torch_pso import ParticleSwarmOptimizer
import copy
# Create Environment
ns = 300
env = RSR(ns, test=False, plot=False)


def sample_uniform_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape) * (param_max - param_min) + param_min
              for k, v in params_prev.items()}
    return params


def sample_local_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape) * (param_max - param_min) + param_min + v
              for k, v in params_prev.items()}
    return params


def plot_simulation(states, actions, control_inputs, ns):
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
  global time_list
  r_list.append(r_tot)
  r_list_i.append(r_tot)
  p_list.append(policy)
  time_list.append((datetime.now() - start_time).total_seconds())
  return r_tot

def policy_training(n_fc1, n_fc2, n_layers,reps,max_iter,n_particles,evals_rs):
  
  
  training_data = np.zeros((2, max_iter*n_particles+evals_rs+30, reps))
  for r_i in range(reps):
    print('Repition:', r_i+1)
    global r_list
    global p_list
    global r_list_i
    global time_list
    global start_time
    r_list = []
    p_list = []
    r_list_i = []
    time_list = []
    start_time = datetime.now()
    policy = Net(n_fc1, n_fc2,activation = torch.nn.ReLU,n_layers = n_layers,deterministic = False)


    # #Training Loop Parameters
    k0     = 7.2e10 # Pre-exponential factor (1/sec)
    UA     = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
    old_swarm  = 1e8
    new_swarm = 0
    tol = 0.01

    env = RSR(ns,test = False,plot= False)

    

    policy_list = np.zeros(max_iter)
    reward_list = np.zeros(max_iter)
    old_swarm = np.zeros(max_iter)
    best_reward = 1e8
    i = 0
    r_list = []
    r_list_i =[]
    p_list  = []

    

    params = policy.state_dict()
    #Random Search
    print('Random search to find good initial policy...')
    max_param = 0.3
    min_param = -1*max_param
    for policy_i in range(evals_rs):
        # sample a random policy
        NNparams_RS  = sample_uniform_params(params,max_param, min_param)
        # consruct policy to be evaluated
        policy.load_state_dict(NNparams_RS)
        # evaluate policy
        r = criterion(policy,ns)
        #Store rewards and policies
        if r < best_reward:
            best_policy = p_list[r_list.index(r)]
            best_reward = r
            init_params= copy.deepcopy(NNparams_RS)
    policy.load_state_dict(init_params)
    #PSO Optimisation paramters
    optim = ParticleSwarmOptimizer(policy.parameters(),
                                  inertial_weight=0.5,
                                  num_particles=n_particles,
                                  max_param_value=max_param,
                                  min_param_value=min_param)
    print('Best reward after random search:', best_reward)
    print('PSO Algorithm...')
    while i < max_iter and abs(best_reward - old_swarm[i]) > tol :
        if i > 0:
          old_swarm[i] = min(r_list_i)
          del r_list_i[:]
        def closure():
            # Clear any grads from before the optimization step, since we will be changing the parameters
            optim.zero_grad()
            return criterion(policy,ns)
        optim.step(closure)
        new_swarm = min(r_list_i)

        if new_swarm < best_reward:
          best_reward = new_swarm
          best_policy = p_list[r_list.index(new_swarm)]
          
          print('New best reward:', best_reward,'iteration:',i+1,'/',max_iter)
        i += 1
    print('Finished optimisation')
    print('Best reward:', best_reward)

    training_data[0,:, r_i] = r_list
   
    training_data[1,:, r_i] = time_list
  return training_data
    





max_iter = 30
evals_rs = 15
n_particles = 5
reps = 3
networks = np.array(([256,256,0],[128,128,0],[64,64,0],[32,32,0]))#np.array(([256,256,1],[128,128,1]))
# td = np.zeros((2,max_iter*n_particles+evals_rs+30,reps,networks.shape[0]))

# for i, net in enumerate(networks):
#   td[:,:,:,i] = policy_training(net[0],net[1],net[2],reps,max_iter,n_particles,evals_rs)


#np.save('training_data_2layers_2602.npy',td)
tdbig = np.load('training_data_2202.npy')
tdsmall = np.load('training_data_2202_64_32.npy')
td_3layers = np.zeros((2,max_iter*n_particles+evals_rs+30,reps,4))
td_3layers[:,:,:,:2] = tdbig
td_3layers[:,:,:,2:] = tdsmall
td_2layers = np.load('training_data_2layers_2602.npy')
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'tab:blue', 'tab:orange']
labels = ['256','128','64','32']


def learning_curve(td):
  n_networks = 4
  # plt.rcParams['text.usetex'] = 'True'
  # episodic learning curve
  fig, ax = plt.subplots(1,1)
  ep = np.linspace(0, len(td[0,:,0,0]), len(td[0,:,0,0]))
  for n_i in range(1,3):
    ax.plot(ep,np.median(td[0,:,:,n_i]*-1, axis = 1), color = colors[n_i], label = '2 layers '+labels[n_i])
   # ax.fill_between(ep,np.min(td[0,:,:,n_i]*-1, axis = 1),np.max(td[0,:,:,n_i]*-1,axis = 1), alpha = 0.2, edgecolor = 'none',color = colors[n_i])
  ax.set_xlabel('Episodes')
  ax.set_ylabel('Reward')
  ax.set_ylim(-500,0)
  ax.set_title('Episodic Learning Curve')
  ax.legend()
  #fig.savefig('2_Layers_Eps.pdf')
  plt.show()
  fig, ax = plt.subplots(1,1)
  t = np.linspace(0,np.max(td[1,:,:,:]),len(td[0,:,0,0]))

  for n_i in range(1,3):
    ax.plot(np.median(td[1,:,:,n_i],axis= 1),np.median(td[0,:,:,n_i]*-1, axis = 1), color = colors[n_i], label = '2 layers '+ labels[n_i])
    #ax.fill_between(np.median(td[1,:,:,n_i],axis= 1),np.min(td[0,:,:,n_i]*-1, axis = 1),np.max(td[0,:,:,n_i]*-1,axis = 1), alpha = 0.2, edgecolor = 'none',color = colors[n_i])
  ax.set_xlabel('Time (s)')
  ax.set_ylim(-500,0)
  ax.set_ylabel('Reward')
  ax.set_title('Time Learning Curve')
  ax.legend()
  #fig.savefig('2_Layers_Time.pdf')
  plt.show()

for i in range(4):
    print(f'2 layers & {labels[i]} neurons per fc layer had a final return of {np.median(td_2layers[0,-1,:,i])}')
    print(f'3 layers & {labels[i]} neurons per fc layer had a final return of {np.median(td_3layers[0,-1,:,i])})')
# time learning curve
# learning_curve(td)