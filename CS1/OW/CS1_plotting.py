import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
  def __init__(self, n_fc1, n_fc2, activation,n_layers,**kwargs):
    super(Net, self).__init__()

    # Unpack the dictionary
    self.args     = kwargs
    self.dtype    = torch.float
    self.use_cuda = torch.cuda.is_available()
    self.device   = torch.device("cpu")

    self.input_size = 6 #State size: Ca, T, Ca setpoint and T setpoint
    self.output_sz  = 7 #Output size: Reactor Ks size
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

  def forward(self, x):

    x = x.float()
    y           = self.act(self.hidden1(x))
    y           = self.act(self.hidden2(y))        
    mu = self.output_mu(y)
    log_std = self.output_std(y) 
    dist = torch.distributions.Normal(mu, log_std.exp()+ 1e-6)
    out = dist.sample()                                         
    y = F.tanh(out) #[-1,1]
   

    return y

best_policy = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1 )
best_policy.load_state_dict(torch.load('best_policy_0502.pth'))
ns = 120
env = reactor_class(test = True,ns = 120)
Ca_des = [0.87 for i in range(int(2*ns/5))] + [0.91 for i in range(int(ns/5))] + [0.85 for i in range(int(2*ns/5))]                     
T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]

model = SAC.load('SAC_0602')
reps = 10
Ca_eval_PG = np.zeros((ns,reps))
T_eval_PG = np.zeros((ns,reps))
Tc_eval_PG = np.zeros((ns,reps))
ks_eval_PG = np.zeros((7,ns,reps))

SP = np.array([Ca_des,T_des])
for r_i in range(reps):
  s_norm,_ = env.reset()
  s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
  Ca_eval_PG[0,r_i] = s[0]
  T_eval_PG[0,r_i] = s[1]
  Tc_eval_PG[0,r_i] = 300.0
  a_policy = model.predict(s_norm,deterministic=True)[0]
  
  a_sim = a_policy
  for ks_i in range(0,3):
      a_sim[ks_i] = (a_sim[ks_i])
      
  for ks_i in range(3,6):
      a_sim[ks_i] = (a_sim[ks_i])
       
  a_sim[6] = (a_sim[ks_i]) + 293
  ks_eval_PG[:,0,r_i] = a_sim
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
    ks_eval_PG[:,i,r_i] = a_sim
    
    
    a_copy = copy.deepcopy(a_sim)
    s_norm, r, done, info,_ = env.step(a_policy)
    a_sim = a_copy
    r_tot += r
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval_PG[i,r_i] = s[0]
    T_eval_PG[i,r_i] = s[1]
    
    Tc_eval_PG[i,r_i] = env.u_history[-1]

Ca_eval_EA = np.zeros((ns,reps))
T_eval_EA = np.zeros((ns,reps))
Tc_eval_EA = np.zeros((ns,reps))
ks_eval_EA = np.zeros((7,ns,reps))

SP = np.array([Ca_des,T_des])
for r_i in range(reps):
  s_norm,_ = env.reset()
  s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
  Ca_eval_EA[0,r_i] = s[0]
  T_eval_EA[0,r_i] = s[1]
  Tc_eval_EA[0,r_i] = 300.0
  a_policy = best_policy(torch.tensor(s_norm))
  a_policy = (a_policy+1)/2 # [-1,1] -> [0,1]
  a_sim = a_policy
  for ks_i in range(0,3):
      a_sim[ks_i] = (a_sim[ks_i])
      
  for ks_i in range(3,6):
      a_sim[ks_i] = (a_sim[ks_i])
       
  a_sim[6] = (a_sim[ks_i]) + 293
  ks_eval_EA[:,0,r_i] = a_sim
  r_tot = 0
  for i in range(1,ns):
    
    if i % 5 == 0:
      a_policy = best_policy(torch.tensor(s_norm))
      a_policy = (a_policy+1)/2 # [-1,1] -> [0,1]
      a_sim = a_policy
      for ks_i in range(0,3):
          a_sim[ks_i] = (a_sim[ks_i])*1
          
      for ks_i in range(3,6):
          a_sim[ks_i] = (a_sim[ks_i])*1
        
      a_sim[6] = (a_sim[ks_i]) + 293
    ks_eval_EA[:,i,r_i] = a_sim
    
    
    a_copy = copy.deepcopy(a_sim)
    s_norm, r, done, info,_ = env.step(a_policy)
    a_sim = a_copy
    r_tot += r
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval_EA[i,r_i] = s[0]
    T_eval_EA[i,r_i] = s[1]
    
    Tc_eval_EA[i,r_i] = env.u_history[-1]

def plot_simulation_comp(Ca_dat_PG, T_dat_PG, Tc_dat_PG,ks_eval_PG,Ca_dat_EA, T_dat_EA, Tc_dat_EA,ks_eval_EA,SP,ns):
  plt.rcParams['text.usetex'] = 'True'
  t = np.linspace(0,25,ns)
  fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  labels = ['$Ca_{k_p}$','$Ca_{k_i}$','$Ca_{k_d}$','$T_{k_p}$','$T_{k_i}$','$T_{k_d}$']
  col = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']
  col_fill = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']

  axs[0].plot(t, np.median(Ca_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'SAC')
  axs[0].plot(t,np.median(Ca_dat_EA,axis=1), color = 'tab:blue', lw=1, label = 'EA')
  axs[0].fill_between(t, np.min(Ca_dat_PG,axis=1), np.max(Ca_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')
  axs[0].fill_between(t, np.min(Ca_dat_EA,axis=1), np.max(Ca_dat_EA,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
  axs[0].step(t, Ca_des, '--', lw=1.5, color='black')
  axs[0].set_ylabel('Ca (mol/m$^3$)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))

  axs[1].plot(t, np.median(T_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'SAC')
  axs[1].plot(t,np.median(T_dat_EA,axis=1), color = 'tab:blue', lw=1, label = 'EA')
  axs[1].fill_between(t, np.min(T_dat_PG,axis=1), np.max(T_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')
  axs[1].fill_between(t, np.min(T_dat_EA,axis=1), np.max(T_dat_EA,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
  axs[1].step(t, T_des, '--', lw=1.5, color='black')
  axs[1].set_ylabel('Ca (mol/m$^3$)')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))
   
  axs[2].step(t, np.median(Tc_dat_PG,axis=1), 'r--', where = 'post',lw=1, label = 'SAC')
  axs[2].step(t, np.median(Tc_dat_EA,axis=1), 'b--', where= 'post',lw=1, label = 'EA')
  axs[2].set_ylabel('Cooling T (K)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))
  plt.show()

  fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  axs[0].set_title('EA PID Parameters')
  for ks_i in range(3):
    axs[0].step(t, np.median(ks_eval_EA[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = labels[ks_i])
    # plt.gca().fill_between(t, np.min(ks_eval_EA[ks_i,:,:],axis=1), np.max(ks_eval_EA[ks_i,:,:],axis=1),
    #                         color=col_fill[ks_i], alpha=0.2)
  axs[0].set_ylabel('Ca PID Parameter (EA)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))
  i = [3,4,5]
  for ks_i in i:
    axs[1].step(t, np.median(ks_eval_EA[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = labels[ks_i])
    # axs[1].gca().fill_between(t, np.min(ks_eval_EA[ks_i,:,:],axis=1), np.max(ks_eval_EA[ks_i,:,:],axis=1),
    #                         color=col_fill[ks_i], alpha=0.2)
  axs[1].set_ylabel('T PID Parameter (EA)')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))

    
  axs[2].step(t, np.median(ks_eval_EA[6,:,:],axis=1), 'c-',where = 'post', lw=1,label = 'Baseline Ks')
  # axs[2].gca().fill_between(t, np.min(ks_eval_EA[6,:,:],axis=1), np.max(ks_eval_EA[6,:,:],axis=1),
  #                           color='c', alpha=0.2)
  axs[2].set_ylabel('Baseline PID Parameter (EA)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))
  plt.show()
  

  fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  axs[0].set_title('SAC PID Parameters')
  for ks_i in range(3):
    axs[0].step(t, np.median(ks_eval_PG[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = labels[ks_i])
    # plt.gca().fill_between(t, np.min(ks_eval_PG[ks_i,:,:],axis=1), np.max(ks_eval_PG[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)
                            
  axs[0].set_ylabel('Ca PID Parameter (SAC)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))

 
  i = [3,4,5]
  for ks_i in i:
    axs[1].step(t, np.median(ks_eval_PG[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = labels[ks_i])
    # axs[1].gca().fill_between(t, np.min(ks_eval_PG[ks_i,:,:],axis=1), np.max(ks_eval_PG[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)
                            
  axs[1].set_ylabel('T PID Parameter (SAC)')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))

 
  axs[2].step(t, np.median(ks_eval_PG[6,:,:],axis=1), 'c-',where = 'post', lw=1,label = 'Baseline Ks')
  # axs[2].gca().fill_between(t, np.min(ks_eval_PG[6,:,:],axis=1), np.max(ks_eval_PG[6,:,:],axis=1),color='c', alpha=0.2)
                            
  axs[2].set_ylabel('Baseline PID Parameter (SAC)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))

  plt.tight_layout()
  plt.show()

plot_simulation_comp(Ca_eval_PG, T_eval_PG, Tc_eval_PG,ks_eval_PG,Ca_eval_EA, T_eval_EA, Tc_eval_EA,ks_eval_EA,SP,ns)