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
def sample_uniform_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min \
              for k, v in params_prev.items()}
    return params

def sample_local_params(params_prev, param_max, param_min):
    params = {k: torch.rand(v.shape)* (param_max - param_min) + param_min + v \
              for k, v in params_prev.items()}
    return params
# Create a gym environment
class RSR(gym.Env):
  def __init__(self,ns,test, plot = False):
    self.i = 0
    self.ns = ns 
    self.T  = 100
    self.Nx = 12
    self.plot = plot  
    self.test = test
    self.info = {'control_in':0,'PID_Action':0,'state':0}
    # Time Interval (min)
    self.t = np.linspace(0,self.T,self.ns)
    # Casadi Model
    self.sym_x = self.gen_casadi_variable(12, "x")
    self.sym_u = self.gen_casadi_variable(5, "u")    
    large_scale_ode = self.large_scale_ode
    self.casadi_sym_model = self.casadify(large_scale_ode, self.sym_x, self.sym_u)
    self.casadi_model_func = self.gen_casadi_function([self.sym_x, self.sym_u],[self.casadi_sym_model],
                                                    "model_func", ["x","u"], ["model_rhs"])
    # Observation Space
    self.observation_space = spaces.Box(low =np.array([10,10,10,10,10,10,20,20,20,1]) , high = np.array([30,30,30,30,30,30,21,21,21,3]))
    # Action Space
    self.action_space = spaces.Box(low = -1, high = 1, shape = (16,))
    self.action_space_unnorm = spaces.Box(low = np.array([8,8,0.67,8]), high = np.array([47,47,3,45]))
    self.PID_space = spaces.Box(low = np.array([8,2,0,self.action_space_unnorm.low[0]-1,8,2,0.1,self.action_space_unnorm.low[1]-1,8,0,0.1,self.action_space_unnorm.low[2]-1,12,0,0,self.action_space_unnorm.low[3]+2]),
                                 high = np.array([14,8,0.4,self.action_space_unnorm.low[0]+1,14,8,0.4,self.action_space_unnorm.low[1]+1,14,5,0.4,self.action_space_unnorm.low[2]+1,18,5,0.4,self.action_space_unnorm.low[3]+4] ),)
    
    self.SP = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))],[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))],[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]])
    self.x0 = np.array([20, 0.8861, 0.1082, 0.0058, 22, 0.8861, 0.1082, 0.0058, 20, 0.1139, 0.7779, 0.1082,self.SP[0,0],self.SP[1,0],self.SP[2,0]])
    self.e_history = []

  def reset(self, seed = None):
    self.F0 = np.array([1,3])
    self.state_hist = np.zeros((self.ns+1,self.x0.shape[0]))
    self.state = self.x0
    self.info['state'] = self.state[:self.Nx]
    self.i = 0
    self.done = False
    rl_state = [self.state[i] for i in [0, 4, 8,0, 4, 8,12,13,14]]
    rl_state.append(self.F0[0])
    self.state_hist[self.i] = self.state
    self.norm_state = (rl_state- self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    return self.norm_state, {}
  def step(self, action):
    action  = (action + 1)/2
    if self.i % 5 == 0:
      self.action = action * (self.PID_space.high - self.PID_space.low) + self.PID_space.low
    try:
      self.state[:self.Nx] = self.integrate(self.action)
      
    except:
      print('Integration Error')
      rew = -1e5
      self.done = True
    rew = self.reward(self.state[:self.Nx])
    self.info['PID_Action'] = self.action
    self.i += 1
    
    if self.i == self.ns:
      self.done = True
    self.state_hist[self.i] = self.state
    rl_state = np.array([self.state[0],self.state[4],self.state[8],self.state_hist[self.i-1][0],self.state_hist[self.i-1][4],self.state_hist[self.i-1][8],self.state[12],self.state[13],self.state[14]])
    if self.i < self.ns/3:
      rl_state = np.append(rl_state,1)
    else: 
      rl_state = np.append(rl_state,3)
    
    self.norm_state = (rl_state- self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    return self.norm_state,rew,self.done,False,self.info

  
  def reward(self, state):
    
    state = [state[i] for i in [0, 4, 8]]
    error = np.sum((self.SP[:,self.i] - state)**2)
    
    return -error
     
  def integrate(self, PID_gains):
    
    state = self.state[:self.Nx]
    Holdups = [state[i] for i in [0, 4, 8]]
    self.e = np.array([Holdups])-self.SP[:,self.i] 
   
    uk = np.zeros(5)

    uk[0] = self.PID_F_R(PID_gains[0:4])
    uk[1] = self.PID_F_M(PID_gains[4:8])
    uk[2] = self.PID_B(PID_gains[8:12])
    uk[3] = self.PID_D(PID_gains[12:16])
    if self.i < self.ns/3:
      uk[4] = 1.5
    else: 
      uk[4] = 2
    self.info['control_in'] = uk

    self.e_history.append(self.e[0]) 
    plant_func = self.casadi_model_func
    discretised_plant = self.discretise_model(plant_func,self.T/self.ns) 
  
    xk = self.state[:self.Nx]
    Fk = discretised_plant(x0=xk, p=uk)
    self.info['state'] = Fk['xf'].full().flatten()   
    return Fk['xf'].full().flatten()
  
  def PID_F_R(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1]    
    k_d = PID_gains[2]
    k_b = PID_gains[3] 
    e_history = np.array(self.e_history)
    e = self.e[0]
    if self.i == 0:
      e_history = np.zeros((1,3))
    
    u = k_p *e[0] + k_i *sum(e_history[:,0]) + k_d *(e[0]-e_history[-1,0])
  
    u += k_b
  
    u = np.clip(u, self.action_space_unnorm.low[0], self.action_space_unnorm.high[0])

    return u
  
  def PID_F_M(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1]    
    k_d = PID_gains[2]
    k_b = PID_gains[3] 
    e_history = np.array(self.e_history)
    e = self.e[0]
    
    if self.i == 0:
      e_history = np.zeros((1,3))
    u = k_p *e[1] + k_i *sum(e_history[:,1]) + k_d *(e[1]-e_history[-1,1])
    u += k_b
    u = np.clip(u, self.action_space_unnorm.low[1], self.action_space_unnorm.high[1])
  
    return u
     
  def PID_B(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1]    
    k_d = PID_gains[2]
    k_b = PID_gains[3] 
    e_history = np.array(self.e_history)
    e = self.e[0]
    if self.i == 0:
      e_history = np.zeros((1,3))
    u = k_p *e[2] + k_i *sum(e_history[:,2]) + k_d *(e[2]-e_history[-1,2])
    u += k_b
    u = np.clip(u, self.action_space_unnorm.low[2], self.action_space_unnorm.high[2])
    return u
    
  def PID_D(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1]    
    k_d = PID_gains[2]
    k_b = PID_gains[3] 
    e_history = np.array(self.e_history)
    e = self.e[0]
    if self.i == 0:
      e_history = np.zeros((1,3))
    u = k_p *e[2] + k_i *sum(e_history[:,2]) + k_d *(e[2]-e_history[-1,2])
    
    u += k_b
    u = np.clip(u, self.action_space_unnorm.low[3], self.action_space_unnorm.high[3])
    return u
  def casadify(self, model, sym_x, sym_u):
    """
    Given a model with Nx states and Nu inputs and returns rhs of ode,
    return casadi symbolic model (Not function!)
    
    Inputs:
        model - model to be casidified i.e. a list of ode rhs of size Nx
        
    Outputs:
        dxdt - casadi symbolic model of size Nx of rhs of ode
    """

    dxdt = model(sym_x, sym_u)
    dxdt = vertcat(*dxdt) #Return casadi list of size Nx

    return dxdt


  def gen_casadi_variable(self, n_dim, name = "x"):
        """
        Generates casadi symbolic variable given n_dim and name for variable
        
        Inputs:
            n_dim - symbolic variable dimension
            name - name for symbolic variable
            
        Outputs:
            var - symbolic version of variable
        """

        var = SX.sym(name, n_dim)

        return var

  def gen_casadi_function(self, casadi_input, casadi_output, name, input_name=[], output_name=[]):
        """
        Generates a casadi function which maps inputs (casadi symbolic inputs) to outputs (casadi symbolic outputs)
        
        Inputs:
            casadi_input - list of casadi symbolics constituting inputs
            casadi_output - list of casadi symbolic output of function
            name - name of function
            input_name - list of names for each input
            output_name - list of names for each output
        
        Outputs:
            casadi function mapping [inputs] -> [outputs]
        
        """

        function = Function(name, casadi_input, casadi_output, input_name, output_name)

        return function
    
  def discretise_model(self, casadi_func, delta_t):
        """
        Input:
            casadi_func to be discretised
        
        Output:
            discretised casadi func
        """
        x = SX.sym("x", 12)
        u = SX.sym("u", 5)
        xdot = casadi_func(x, u)
        dae = {'x':x, 'p':u, 'ode':xdot} 
        t0 = 0
        tf = delta_t
        discrete_model = integrator('discrete_model', 'cvodes', dae,t0,tf)

        return discrete_model
        

    
  def large_scale_ode(self,x, u):

      #Section 3.2 (Example 2) from https://www.sciencedirect.com/science/article/pii/S0098135404001784
      #This is a challenging control problem as the system can exhibit a severe snowball effect (Luyben, Tyr ́eus, & Luyben, 1999) if

      #Parameters
      rho = 1 #Liquid density
      alpha_1 = 90 #Volatility see: http://www.separationprocesses.com/Distillation/DT_Chp01-3.htm#:~:text=VLE%3A%20Relative%20Volatility&text=In%20order%20to%20separate%20a,is%20termed%20the%20relative%20volatility.
      k_1 = 0.0167 #Rate constant
      k_2 = 0.0167 #Rate constant
      A_R = 5 #Vessel area
      A_M = 10 #Vessel area
      A_B = 5 #Vessel area
      x1_O = 1.00

      ###Model Equations###

      ##States##
      #H_R - Liquid level of reactor
      #x1_R - Molar liquid fraction of component 1 in reactor
      #x2_R - Molar liquid fraction of component 2 in reactor
      #x3_R - Molar liquid fraction of component 3 in reactor
      #H_M - Liquid level of storage tank
      #x1_M - Molar liquid fraction of component 1 in storage tank
      #x2_M - Molar liquid fraction of component 2 in storage tank
      #x3_M - Molar liquid fraction of component 3 in storage tank
      #H_B - Liquid level of flash tank
      #x1_B - Molar liquid fraction of component 1 in bottoms
      #x2_B - Molar liquid fraction of component 2 in bottoms
      #x3_B - #Molar liquid fraction of component 3 in bottoms

      H_R, x1_R, x2_R, x3_R, H_M, x1_M, x2_M, x3_M, H_B, x1_B, x2_B, x3_B = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11]

      ##Inputs##
      #F_O - Reactor input flowrate
      #F_R - Reactor outlet flowrate
      #F_M - Storage tank outlet flowrate
      #B - Bottoms flowrate
      #D - Distillate flowrate  
    
      F_R, F_M, B, D,F_O =  u[0], u[1], u[2], u[3],u[4]

      #Calculate distillate composition (only component 1 and 2 are volatile and component 3 is not
      #while component 1 is 90 times more volatile than component 2)

      x1_D = ((x1_B * alpha_1) / (1 - x1_B + x1_B * alpha_1))
      x2_D = 1 - x1_D

      dxdt = [
          (1/(rho*A_R)) * (F_O + D - F_R),
          ((F_O*(x1_O - x1_R) + D*(x1_D - x1_R))/(rho*A_R*H_R)) - k_1 * x1_R,
          ((-F_O * x2_R + D * (x2_D - x2_R))/(rho*A_R*H_R)) + k_1 * x1_R - k_2 * x2_R,
          ((-x3_R*(F_O + D))/(rho*A_R*H_R)) + k_2 * x2_R,
          (1/(rho*A_M)) * (F_R - F_M),
          ((F_R)/(rho*A_M*H_M))*(x1_R - x1_M),
          ((F_R)/(rho*A_M*H_M))*(x2_R - x2_M),
          ((F_R)/(rho*A_M*H_M))*(x3_R - x3_M),
          (1/(rho*A_B))*(F_M - B - D),
          (1/(rho*A_B*H_B))*(F_M*(x1_M-x1_B) - D*(x1_D - x1_B)),
          (1/(rho*A_B*H_B))*(F_M*(x2_M-x2_B) - D*(x2_D - x2_B)),
          (1/(rho*A_B*H_B))*(F_M*(x3_M-x3_B) + D*(x3_B))
          ]

      return dxdt
  
def plot_simulation(states,actions, control_inputs,ns):
    plt.rcParams['text.usetex'] = 'True'
    
    actions = np.array(actions)
    control_inputs = np.array(control_inputs)
    SP = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]]).reshape(ns,1)
    SP_M = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]]).reshape(ns,1)
    data = [np.array(states)[:,i] for i in [0,4,8,1,2,3,5,6,7,9,10,11]]
   
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
        axs[0,0].plot(t,data[i], color=colors[i], label=labels[i])
        axs[0,0].set_ylabel('Vessel Holdup')
    #axs[0,0].step(t, SP, 'k--', where  = 'post',label='SP$_R$ \& SP$_B$')
    axs[0,0].step(t, SP_M, 'k-.', where  = 'post',label='SP$_M$')
   
    axs[0,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3,frameon=False)
    axs[0,0].set_ylim(20.5, 21.5)
    
    #Component Fraction Plots
    for i in range(3, len(data)):
        axs[0,1].plot(t,data[i], color=colors[i], label=labels[i])
    axs[0,1].set_ylabel('Liquid Component Fraction')
    axs[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.45),
          ncol=3,frameon=False)
    #PID Action Plots
    for i in range(0,4):
        axs[1,0].step(t, actions[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
    axs[1,0].set_ylabel('$F_R$ PID Action')
    axs[1,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)
    for i in range(4,8):
        axs[1,1].step(t, actions[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
    axs[1,1].set_ylabel('$F_M$ PID Action')
    axs[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)
    for i in range(8,12):
        axs[2,0].step(t, actions[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
    axs[2,0].set_ylabel('$B$ PID Action')
    axs[2,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)
    
    for i in range(12,16):
        axs[2,1].step(t, actions[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label=PID_labels[i % len(PID_labels)])
    axs[2,1].set_ylabel('$D$ PID Action')
    axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=4,frameon=False)

    for i in range(2):
        axs[3,0].step(t, control_inputs[:,i], where='post', linestyle='dashed', color=colors[i % len(colors)], label=control_labels[i])
    axs[3,0].set_ylabel('Flowrate (h$^{-1}$)')
    axs[3,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=10,frameon=False)
   
 
    for i in range(2,4):
        axs[3,1].step(t, control_inputs[:,i], where='post', linestyle='dashed', color=colors[i % len(colors)], label=control_labels[i])
    axs[3,1].set_ylabel('Flowrate (h$^{-1}$)')
    axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
          ncol=10,frameon=False)
    
    plt.subplots_adjust(hspace=0.5)
   
    plt.show()

def rollout(ns,policy):
    env = RSR(ns,test=False,plot=False)
    s,_ = env.reset()
    done = False
    states = []
    actions = []
    rewards = []
    controls = []
    while not done:
        action = policy.predict(s)[0]
        state, reward, done, _, control = env.step(action)
        #un normalise state
        states.append(control['state'])
        actions.append(control['PID_Action'])
        rewards.append(reward)
        controls.append(control['control_in'])
    return states, actions, rewards,controls


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

env = RSR(260,test = False,plot= False)
ns = 260

def rollout_DFO(ns,policy):
    env = RSR(ns,test=False,plot=False)
    s,_ = env.reset()
    done = False
    states = []
    actions = []
    rewards = []
    controls = []
    while not done:
        action = policy.predict(torch.tensor(s))[0]
       
        s, reward, done, _,control = env.step(action)
        #un normalise state
        states.append(control['state'])
      
        actions.append(control['PID_Action'])
        rewards.append(reward)
        controls.append(control['control_in'])
    
    return states, actions, rewards,controls

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
  s,a,r,c = rollout_DFO(ns,policy)
  r = np.array(r)
  r_tot = -1*np.sum(r)
  
  global r_list
  global p_list
  global r_list_i
  r_list.append(r_tot)
  r_list_i.append(r_tot)
  p_list.append(policy)
  return r_tot
policy = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1,deterministic = False)


#Training Loop Parameters
k0     = 7.2e10 # Pre-exponential factor (1/sec)
UA     = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
old_swarm  = 1e8
new_swarm = 0
tol = 0.01
ns = 300
env = RSR(ns,test = False,plot= False)

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
print('Plotting Best Policy...')
s,a,r,c = rollout_DFO(ns,best_policy)
plot_simulation(s,a,c,ns)

# Ks = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# s,a,r,c = rollout_test(ns,Ks)

# plot_simulation(s,a,c,ns)
# #
# model = SAC('MlpPolicy', env, verbose = 1,learning_rate=1e-5,device='cuda')
# model.learn(total_timesteps=1e4)

# env = RSR(260,test = False,plot= False)
# ns = 260
# s, a,r,c = rollout(ns, model)

# plot_simulation(s,a,c,ns)