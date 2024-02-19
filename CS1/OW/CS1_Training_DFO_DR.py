import torch
import torch.nn.functional as F
from CS1_Model import reactor_class
from torch_pso import ParticleSwarmOptimizer
import copy
import numpy as np
def sample_uniform_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min \
              for k, v in params_prev.items()}
    return params

def sample_local_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min + v \
              for k, v in params_prev.items()}
    return params

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
def criterion(policy,SP,ns,k0,UA,env):
  
  s, _  = env.reset()
  rew = 0
  while True:
    a = policy(torch.tensor(s)).detach().numpy()
    s, r, done,_,_ = env.step(a)
    r = -r #PSO minimises
    rew += r
    if done:
      break 
 
  r_tot = rew
  global r_list
  global p_list
  global r_list_i
  r_list.append(r_tot)
  r_list_i.append(r_tot)
  p_list.append(policy)
  return r_tot

policy = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1 )
env = reactor_class(test = False,ns = 120,DR=True)
#Training Loop Parameters
k0     = 7.2e10 # Pre-exponential factor (1/sec)
UA     = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
old_swarm  = 1e8
new_swarm = 0
tol = 0.01
ns = 120
Ca_des1 = [0.8 for i in range(int(ns/2))] + [0.9 for i in range(int(ns/2))]
T_des1  = [330 for i in range(int(ns/2))] + [320 for i in range(int(ns/2))]

Ca_des2 = [0.7 for i in range(int(ns/2))] + [0.9 for i in range(int(ns/2))]
T_des2  = [340 for i in range(int(ns/2))] + [320 for i in range(int(ns/2))]

Ca_des3 = [0.9 for i in range(int(ns/2))] + [0.8 for i in range(int(ns/2))]
T_des3  = [320 for i in range(int(ns/2))] + [330 for i in range(int(ns/2))]

Ca_des4 = [0.9 for i in range(int(ns/2))] + [0.7 for i in range(int(ns/2))]
T_des4  = [320 for i in range(int(ns/2))] + [340 for i in range(int(ns/2))]

Ca_disturb = [0.8 for i in range(ns)]
T_disturb = [330 for i in range(ns)]
SP = np.array(([Ca_des1,T_des1],[Ca_des2,T_des2],[Ca_des3,T_des3],[Ca_des4,T_des4],[Ca_disturb,T_disturb]),dtype = object)

max_iter = 30

policy_list = np.zeros(max_iter)
reward_list = np.zeros(max_iter)
old_swarm = np.zeros(max_iter)
best_reward = 1e8
i = 0
r_list = []
r_list_i =[]
p_list  =[]

evals_rs = 30

params = policy.state_dict()
#Random Search
print('Random search to find good initial policy...')
for policy_i in range(evals_rs):
    # sample a random policy
    NNparams_RS  = sample_uniform_params(params, 0.1, -0.1)
    # consruct policy to be evaluated
    policy.load_state_dict(NNparams_RS)
    # evaluate policy
    r = criterion(policy,SP,ns,k0,UA,env)
    #Store rewards and policies
    if r < best_reward:
        best_policy = p_list[r_list.index(r)]
        best_reward = r
        init_params= copy.deepcopy(NNparams_RS)
policy.load_state_dict(init_params)
#PSO Optimisation paramters
optim = ParticleSwarmOptimizer(policy.parameters(),
                               inertial_weight=0.5,
                               num_particles=15,
                               max_param_value=0.2,
                               min_param_value=-0.2)
print('Best reward after random search:', best_reward)
print('PSO Algorithm...')
while i < max_iter and abs(best_reward - old_swarm[i]) > tol :
    if i > 0:
      old_swarm[i] = min(r_list_i)
      del r_list_i[:]
    def closure():
        # Clear any grads from before the optimization step, since we will be changing the parameters
        optim.zero_grad()
        return criterion(policy,SP,ns,k0,UA,env)
    optim.step(closure)
    new_swarm = min(r_list_i)

    if new_swarm < best_reward:
      best_reward = new_swarm
      best_policy = p_list[r_list.index(new_swarm)]
      
      print('New best reward:', best_reward,'iteration:',i+1,'/',max_iter)
    i += 1
print('Finished optimisation')
print('Best reward:', best_reward)
print('Saving robust model..')
torch.save(best_policy.state_dict(), 'DFO_best_policy_DR_1902.pth')