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
    self.output_sz  = 4 #Output size: Reactor Ks size
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
best_policy.load_state_dict(torch.load('best_policy_DFO_Vel_002.pth'))
ns = 240
env = reactor_class(test = True,ns = 240,PID_vel=True)

Ca_des = [0.95 for i in range(int(ns/3))] + [0.9 for i in range(int(ns/3))] + [0.85 for i in range(int(ns/3))]  
T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]

model = SAC.load('SAC_Vel_0103')
reps = 10
Ca_eval_PG = np.zeros((ns,reps))
T_eval_PG = np.zeros((ns,reps))
Tc_eval_PG = np.zeros((ns,reps))
ks_eval_PG = np.zeros((4,ns,reps))

SP = np.array([Ca_des,T_des])
for r_i in range(reps):
  s_norm,_ = env.reset()
  s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
  Ca_eval_PG[0,r_i] = s[0]
  T_eval_PG[0,r_i] = s[1]
  Tc_eval_PG[0,r_i] = 300.0
  a_policy = model.predict(s_norm,deterministic=True)[0]
  
  a_sim = a_policy
  a_norm = (a_policy+1)/2 # [-1,1] -> [0,1]
  a_sim = copy.deepcopy(a_norm)
  a_sim[0] = (a_sim[0])*-200
  a_sim[1] = (a_sim[1])*20 + 0.01
  a_sim[2] = (a_sim[2])*10
      
  
  a_sim[3] = (a_sim[3]) + 290
  r_tot = 0
  for i in range(1,ns):
    
    if i % 5 == 0:
      a_policy = model.predict(s_norm,deterministic=True)[0]
    
       # [-1,1] -> [0,1]
      a_norm = (a_policy+1)/2 # [-1,1] -> [0,1]
      a_sim = copy.deepcopy(a_norm)
      a_sim[0] = (a_sim[0])*-200
      a_sim[1] = (a_sim[1])*20 + 0.01
      a_sim[2] = (a_sim[2])*10
          
      
      a_sim[3] = (a_sim[3]) + 290
          
   
    ks_eval_PG[:,i,r_i] = a_sim
    
    
    a_copy = copy.deepcopy(a_sim)
    s_norm, r, done, info,_ = env.step(a_policy)
    a_sim = a_copy
    r_tot += r
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval_PG[i,r_i] = s[0]
    T_eval_PG[i,r_i] = s[1]
    
    Tc_eval_PG[i,r_i] = env.u_history[-1]
print(r_tot)
Ca_eval_EA = np.zeros((ns,reps))
T_eval_EA = np.zeros((ns,reps))
Tc_eval_EA = np.zeros((ns,reps))
ks_eval_EA = np.zeros((4,ns,reps))

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
  a_sim[0] = (a_sim[0])*-200
  a_sim[1] = (a_sim[1])*20 + 0.01
  a_sim[2] = (a_sim[2])*10
      
  
  a_sim[3] = (a_sim[3]) + 290
  ks_eval_EA[:,0,r_i] = a_sim
  r_tot = 0
  for i in range(1,ns):
    
    if i % 5 == 0:
      a_policy = best_policy(torch.tensor(s_norm))
      
      a_norm = (a_policy+1)/2 # [-1,1] -> [0,1]
      a_sim = copy.deepcopy(a_norm)
      a_sim[0] = (a_sim[0])*-200
      a_sim[1] = (a_sim[1])*20 + 0.01
      a_sim[2] = (a_sim[2])*10
          
      
      a_sim[3] = (a_sim[3]) + 290
      
    ks_eval_EA[:,i,r_i] = a_sim
    
    
    a_copy = copy.deepcopy(a_sim)
   
    s_norm, r, done, info,_ = env.step(a_policy)
    a_sim = a_copy
    r_tot += r
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval_EA[i,r_i] = s[0]
    T_eval_EA[i,r_i] = s[1]
    
    Tc_eval_EA[i,r_i] = env.u_history[-1]
print(r_tot)
def rollout(Ks,reps):
  ns = 240
  env = reactor_class(test = True, ns = 240, PID_vel = True)
  Ca_des = [0.95 for i in range(int(ns/3))] + [0.9 for i in range(int(ns/3))] + [0.85 for i in range(int(ns/3))]  
  T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]
  Ca_eval = np.zeros((ns,reps))
  T_eval = np.zeros((ns,reps))
  Tc_eval = np.zeros((ns,reps))
  ks_eval = np.zeros((4,ns,reps))
  r_eval = np.zeros((1,reps))
  SP = np.array([Ca_des,T_des])

 
  
  for r_i in range(reps):
    s_norm,_ = env.reset()
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval[0,r_i] = s[0]
    T_eval[0,r_i] = s[1]
    Tc_eval[0,r_i] = 300.0
    ks_eval[:,0,r_i] = Ks[:4]
    r_tot = 0
    Ks_i = 0
    for i in range(1,ns):
      
      
      ks_eval[:,i,r_i] = Ks[Ks_i*4:(Ks_i+1)*4]
      norm_values = np.array(([-200,0,0,290],[0,20.01,10,303]))
      Ks_norm = copy.deepcopy(Ks)
      Ks_norm = ((Ks_norm - norm_values[0]) / (norm_values[1] - norm_values[0])) * 2 - 1
    
      s_norm, r, done, info,_ = env.step(Ks_norm[Ks_i*4:(Ks_i+1)*4])
     
      r_tot += r
      s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
      Ca_eval[i,r_i] = s[0]
      T_eval[i,r_i] = s[1]
      Tc_eval[i,r_i] = env.u_history[-1]
    r_eval[:,r_i] = r_tot
  r = -1*np.mean(r_tot)
  print(r)
  return Ca_eval, T_eval, Tc_eval, ks_eval
def plot_simulation_comp(Ca_dat_PG, T_dat_PG, Tc_dat_PG,ks_eval_PG,Ca_dat_EA, T_dat_EA, Tc_dat_EA,ks_eval_EA,Ca_dat_const, T_dat_const, Tc_dat_const, ks_eval_const,SP,ns):
  plt.rcParams['text.usetex'] = 'False'
  t = np.linspace(0,25,ns)
  fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  labels = ['$Ca_{k_p}$','$Ca_{k_i}$','$Ca_{k_d}$','$T_{k_p}$','$T_{k_i}$','$T_{k_d}$']
  col = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']
  col_fill = ['tab:orange','tab:red','tab:blue','tab:orange','tab:red','tab:blue']

  axs[0].plot(t, np.median(Ca_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'PG')
  axs[0].plot(t,np.median(Ca_dat_EA,axis=1), color = 'tab:blue', lw=1, label = 'EA')
  axs[0].plot(t,np.median(Ca_dat_const,axis=1), color = 'tab:green', lw=1, label = 'Constant')
  axs[0].fill_between(t, np.min(Ca_dat_PG,axis=1), np.max(Ca_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')
  axs[0].fill_between(t, np.min(Ca_dat_EA,axis=1), np.max(Ca_dat_EA,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
  axs[0].fill_between(t, np.min(Ca_dat_const,axis=1), np.max(Ca_dat_const,axis=1),color = 'tab:green', alpha=0.2,edgecolor  = 'none')
  axs[0].step(t, Ca_des, '--', lw=1.5, color='black')
  axs[0].set_ylabel('Ca (mol/m$^3$)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))

  axs[1].plot(t, np.median(T_dat_PG,axis=1), color = 'tab:red', lw=1, label = 'PG')
  axs[1].plot(t,np.median(T_dat_EA,axis=1), color = 'tab:blue', lw=1, label = 'EA')
  axs[1].plot(t,np.median(T_dat_const,axis=1), color = 'tab:green', lw=1, label = 'Constant')
  axs[1].fill_between(t, np.min(T_dat_PG,axis=1), np.max(T_dat_PG,axis=1),color = 'tab:red', alpha=0.2,edgecolor  = 'none')
  axs[1].fill_between(t, np.min(T_dat_EA,axis=1), np.max(T_dat_EA,axis=1),color = 'tab:blue', alpha=0.2,edgecolor  = 'none')
  axs[1].fill_between(t, np.min(T_dat_const,axis=1), np.max(T_dat_const,axis=1),color = 'tab:green', alpha=0.2,edgecolor  = 'none')
  axs[1].set_ylabel('Temperature (K))')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))
   
  axs[2].step(t, np.median(Tc_dat_PG,axis=1), color = 'tab:red', linestyle = 'dashed' , where = 'post',lw=1, label = 'PG')
  axs[2].step(t, np.median(Tc_dat_EA,axis=1), color = 'tab:blue', linestyle = 'dashed', where= 'post',lw=1, label = 'EA')
  axs[2].step(t, np.median(Tc_dat_const,axis=1), color = 'tab:green', linestyle = 'dashed', where= 'post',lw=1, label = 'Constant')
  axs[2].set_ylabel('Cooling T (K)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))
  # plt.savefig('velocity_vs_pos_states.pdf')
  plt.show()

  fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  axs[0].set_title('EA PID Parameters')
  
  axs[0].step(t, np.median(ks_eval_EA[0,:,:],axis=1), col[0],where = 'post', lw=1,label = labels[0])
    # plt.gca().fill_between(t, np.min(ks_eval_EA[ks_i,:,:],axis=1), np.max(ks_eval_EA[ks_i,:,:],axis=1),
    #                         color=col_fill[ks_i], alpha=0.2)
  axs[0].set_ylabel('Ca PID Parameter (EA)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))
   
  for ks_i in range(1,3):
    axs[1].step(t, np.median(ks_eval_EA[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = labels[ks_i])
    # plt.gca().fill_between(t, np.min(ks_eval_EA[ks_i,:,:],axis=1), np.max(ks_eval_EA[ks_i,:,:],axis=1),
    #                         color=col_fill[ks_i], alpha=1.2)
  axs[1].set_ylabel('Ca PID Parameter (EA)')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))
  
    
  axs[2].step(t, np.median(ks_eval_EA[3,:,:],axis=1), 'c-',where = 'post', lw=1,label = 'Baseline Ks')
  # axs[2].gca().fill_between(t, np.min(ks_eval_EA[6,:,:],axis=1), np.max(ks_eval_EA[6,:,:],axis=1),
  #                           color='c', alpha=0.2)
  axs[2].set_ylabel('Baseline PID Parameter (EA)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))
  # plt.savefig('velocity_vs_pos_ks_pos.pdf')
  plt.show()
  

  fig, axs = plt.subplots(3, 1, figsize=(10, 12))
  axs[0].set_title('PG PID Parameters')

  axs[0].step(t, np.median(ks_eval_PG[0,:,:],axis=1), col[0],where = 'post', lw=1,label = labels[0])
    # plt.gca().fill_between(t, np.min(ks_eval_PG[ks_i,:,:],axis=1), np.max(ks_eval_PG[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)
                            
  axs[0].set_ylabel('Ca PID Parameter (PG)')
  axs[0].set_xlabel('Time (min)')
  axs[0].legend(loc='best')
  axs[0].set_xlim(min(t), max(t))
  for ks_i in range(1,3):
    axs[1].step(t, np.median(ks_eval_PG[ks_i,:,:],axis=1), col[ks_i],where = 'post', lw=1,label = labels[ks_i])
    # plt.gca().fill_between(t, np.min(ks_eval_PG[ks_i,:,:],axis=1), np.max(ks_eval_PG[ks_i,:,:],axis=1),color=col_fill[ks_i], alpha=0.2)
                            
  axs[1].set_ylabel('Ca PID Parameter (PG)')
  axs[1].set_xlabel('Time (min)')
  axs[1].legend(loc='best')
  axs[1].set_xlim(min(t), max(t))



 
  axs[2].step(t, np.median(ks_eval_PG[3,:,:],axis=1), 'c-',where = 'post', lw=1,label = 'Baseline Ks')
  # axs[2].gca().fill_between(t, np.min(ks_eval_PG[6,:,:],axis=1), np.max(ks_eval_PG[6,:,:],axis=1),color='c', alpha=0.2)
                            
  axs[2].set_ylabel('Baseline PID Parameter (velocity form)')
  axs[2].set_xlabel('Time (min)')
  axs[2].legend(loc='best')
  axs[2].set_xlim(min(t), max(t))

  plt.tight_layout()
  # plt.savefig('velocity_vs_pos_ks_vel.pdf')
  plt.show()
Ks_vel = np.load('GS_Global_vel_const.npy')

Ca_dat_const, T_dat_const, Tc_dat_const, ks_eval_const = rollout(Ks_vel, reps = 10)

plot_simulation_comp(Ca_eval_PG, T_eval_PG, Tc_eval_PG,ks_eval_PG,Ca_eval_EA, T_eval_EA, Tc_eval_EA,ks_eval_EA,Ca_dat_const, T_dat_const, Tc_dat_const, ks_eval_const,SP,ns)