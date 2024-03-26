import matplotlib.pyplot as plt
import numpy as np
from CS1_Model import reactor_class
from stable_baselines3 import SAC
import copy
import torch
import torch.nn.functional as F
from scipy.optimize import differential_evolution, minimize
ns = 240
env = reactor_class(test = True, ns = ns, PID_vel = True)
Ca_des =  env.SP[0][0]
T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]
class Net(torch.nn.Module):

  def __init__(self, n_fc1, n_fc2, activation,n_layers,deterministic,**kwargs):
    super(Net, self).__init__()

    # Unpack the dictionary
    self.args     = kwargs
    self.dtype    = torch.float
    self.use_cuda = torch.cuda.is_available()
    self.device   = torch.device("cpu")

    self.input_size = 5 #State size: Ca, T,Ca-,T- , Ca setpoint 
    self.output_sz  = 3 #Output size: Reactor Ks size
    self.n_layers = torch.nn.ModuleList()
    self.hs1        = n_fc1                                    # !! parameters
    self.hs2        = n_fc2                                      # !! parameter
    self.deterministic = deterministic
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
    if self.deterministic:
      y = F.tanh(mu).detach().numpy()

    return y
def rollout(Ks, PID_Form,opt,reps):
  ns = 240
 
  
  T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]
  Ca_eval = np.zeros((ns,reps))
  T_eval = np.zeros((ns,reps))
  Tc_eval = np.zeros((ns,reps))
  ks_eval = np.zeros((3,ns,reps))
  r_eval = np.zeros((1,reps))
 
 
  env = reactor_class(test = True, ns = ns, PID_vel = True)
  Ca_des = env.SP[0][0]
  SP = np.array([Ca_des,T_des])
  x_norm = env.x_norm

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
       if i % 300 == 0:
          Ks_i += 1
      elif PID_Form == 'GS':
        if i % 40 == 0:
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

# Ks_GS = np.load('GS_Global_vel.npy')
# Ca_dat_GS, T_dat_GS, Tc_dat_GS, ks_eval_GS = rollout(Ks_GS, 'GS', opt = False,reps = 10)

reps = 10
Ca_eval_EA = np.zeros((ns,reps))
T_eval_EA = np.zeros((ns,reps))
Tc_eval_EA = np.zeros((ns,reps))
ks_eval_EA = np.zeros((3,ns,reps))
r_eval_EA = np.zeros((1,reps))
SP = np.array([Ca_des,T_des])
env = reactor_class(test = True, ns = ns, PID_vel = True)
best_policy = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1,deterministic=True)
best_policy.load_state_dict(torch.load('best_policy_DFO_Vel_002.pth'))
for r_i in range(reps):
  s_norm,_ = env.reset()
  s =  s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
  Ca_eval_EA[0,r_i] = s[0]
  T_eval_EA[0,r_i] = s[1]
  Tc_eval_EA[0,r_i] = 300.0
  
  a_policy = best_policy(torch.tensor(s_norm))
  x_norm = env.x_norm
  Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
      
  
  ks_eval_EA[:,0,r_i] = Ks_norm
  r_tot = 0
  for i in range(1,ns):
    if i % 5 == 0:
      a_policy = best_policy(torch.tensor(s_norm))
      
    Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
    ks_eval_EA[:,i,r_i] = Ks_norm
    
   
    s_norm, r, done, info,_ = env.step(a_policy)
   
    r_tot += r
    s =  s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval_EA[i,r_i] = s[0]
    T_eval_EA[i,r_i] = s[1]
    
    Tc_eval_EA[i,r_i] = env.u_history[-1]
  r_eval_EA[:,r_i] = r_tot
print('EA-RL (reward): ',np.round(-1*np.mean(r_eval_EA),2))
ISE_EA = np.sum((Ca_des - np.median(Ca_eval_EA,axis=1))**2)
print('EA-RL (ISE): ',np.round(ISE_EA,3))


model = SAC.load('SAC_Vel_0403_newSP')
reps = 10
Ca_eval_PG = np.zeros((ns,reps))
T_eval_PG = np.zeros((ns,reps))
Tc_eval_PG = np.zeros((ns,reps))
ks_eval_PG = np.zeros((3,ns,reps))
r_eval_PG = np.zeros((1,reps))
SP = np.array([Ca_des,T_des])
for r_i in range(reps):
  s_norm,_ = env.reset()
  s =  s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
  Ca_eval_PG[0,r_i] = s[0]
  T_eval_PG[0,r_i] = s[1]
  Tc_eval_PG[0,r_i] = 300.0
  a_policy = model.predict(s_norm,deterministic=True)[0]

  a_sim = a_policy
  x_norm = env.x_norm
  Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
      

  ks_eval_PG[:,0,r_i] = Ks_norm
      
  
  r_tot = 0
  for i in range(1,ns):
    if i % 5 == 0:
      a_policy = model.predict(s_norm,deterministic=True)[0]
          
      
    a_copy = copy.deepcopy(a_sim)
    s_norm, r, done, _,info = env.step(a_policy)
    Ks_norm = ((a_policy + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
    ks_eval_PG[:,i,r_i] = Ks_norm
    r_tot += r
    s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
    Ca_eval_PG[i,r_i] = s[0]
    T_eval_PG[i,r_i] = s[1]
    
    Tc_eval_PG[i,r_i] = env.u_history[-1]
  r_eval_PG[:,r_i] = r_tot
print('PG-RL (reward): ', np.round(-1*np.mean(r_eval_PG),2))
ISE_PG = np.sum((Ca_des - np.median(Ca_eval_PG,axis=1))**2)
print('PG-RL (ISE): ',np.round(ISE_PG,3))

Ks_GS = np.load('GS_6.npy')
Ca_dat_GS, T_dat_GS, Tc_dat_GS, ks_eval_GS = rollout(Ks_GS, 'GS', opt = False,reps = 10)

Ks_const = np.load('GS_const.npy')
Ca_dat_const, T_dat_const, Tc_dat_const, ks_eval_const = rollout(Ks_const, 'const', opt = False,reps = 10)
plot_simulation_comp(Ca_eval_EA, T_eval_EA, Tc_eval_EA,ks_eval_EA,Ca_dat_const, T_dat_const, Tc_dat_const,ks_eval_const,SP,ns)