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

original_policy = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1 )
DR_policy = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1 ) #TODO: Make deterministic for testing!
original_policy.load_state_dict(torch.load('best_policy_DFO_2002.pth'))
DR_policy.load_state_dict(torch.load('DFO_best_policy_DR_1902.pth'))

env = reactor_class(test = True,ns = 120,robust_test=True)
ns = 120

def rollout(ns,reps,Ca_des,T_des,policy):
    Ca_eval_EA = np.zeros((ns,reps))
    T_eval_EA = np.zeros((ns,reps))
    Tc_eval_EA = np.zeros((ns,reps))
    ks_eval_EA = np.zeros((7,ns,reps))
    reward = np.zeros(reps)
    SP = np.array([Ca_des,T_des])
    for r_i in range(reps):
      s_norm,_ = env.reset()
      s = s_norm*(env.observation_space.high - env.observation_space.low) + env.observation_space.low
      Ca_eval_EA[0,r_i] = s[0]
      T_eval_EA[0,r_i] = s[1]
      Tc_eval_EA[0,r_i] = 300.0
      a_policy = policy(torch.tensor(s_norm))
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
          a_policy = policy(torch.tensor(s_norm))
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
      reward[r_i] = r_tot
    return Ca_eval_EA, T_eval_EA, Tc_eval_EA, ks_eval_EA, reward

Ca_des = [0.87 for i in range(int(2*ns/5))] + [0.91 for i in range(int(ns/5))] + [0.85 for i in range(int(2*ns/5))]                     
T_des  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]
Ca_eval_orig, T_eval_orig, Tc_eval_orig, ks_eval_orig,reward_orig  = rollout(ns,200,Ca_des,T_des,original_policy)

Ca_eval_DR, T_eval_DR, Tc_eval_DR, ks_eval_DR, reward_DR  = rollout(ns,200,Ca_des,T_des,DR_policy)


def hist_plot(reward_DR,reward_orig):
  plt.rcParams['text.usetex'] = 'True'
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.grid(True)
  ax.set_axisbelow(True)  # Draw grid lines behind other graph elements
  ax.hist(reward_DR, bins=30, alpha=0.7, label='DR Policy', color='tab:red',density = True)
  ax.hist(reward_orig, bins=30, alpha=0.7, label='Original Policy', color='tab:blue',density = True)
  ax.set_xlabel('Return', fontsize=14)
  ax.set_ylabel('Density', fontsize=14)
  ax.set_title('Histogram of Return', fontsize=16)
  ax.legend(fontsize=12)
  plt.show()
print('Mean Reward DR:',np.mean(reward_DR),'Mean Reward Original',np.mean(reward_orig))
print('std Reward DR:',np.std(reward_DR),'std Reward Original',np.std(reward_orig))
hist_plot(reward_DR,reward_orig)