import gymnasium as gym
from gymnasium import spaces 
from casadi import *
import numpy as np
import copy
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
    self.observation_space = spaces.Box(low = 0,high=1, shape = (10,))
    self.observation_space_actual = spaces.Box(low =np.array([20,20,20,20,20,20,20,20,20,1]) , high = np.array([21.5,21.5,21.5,22,22,22,21,21,21,3.2]))
    # Action Space
    self.action_space = spaces.Box(low = -1, high = 1, shape = (16,))
    self.action_space_unnorm = spaces.Box(low = np.array([8,8,0.67,8]), high = np.array([47,47,3,45]))
    self.PID_space = spaces.Box(low = np.array([8,2,0,self.action_space_unnorm.low[0]-1,8,2,0.1,self.action_space_unnorm.low[1]-1,9,0,0.1,self.action_space_unnorm.low[2]-1,12,0,0,self.action_space_unnorm.low[3]+2]),
                                 high = np.array([14,8,0.4,self.action_space_unnorm.low[0]+1,14,8,0.4,self.action_space_unnorm.low[1]+1,10,5,0.4,self.action_space_unnorm.low[2]+1,18,5,0.4,self.action_space_unnorm.low[3]+4] ),)
    
    self.SP = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))],[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))],[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]])
    self.x0 = copy.deepcopy(np.array([20.5, 0.8861, 0.1082, 0.0058, 21.5, 0.8861, 0.1082, 0.0058, 20.5, 0.1139, 0.7779, 0.1082,self.SP[0,0],self.SP[1,0],self.SP[2,0]]))
    self.e_history = []

  def reset(self, seed = None):
    if self.test:
      self.F0 = np.array([[1,2.7]])
    else:
      self.F0 = np.array(([1.05,2.4],[1,2.6],[1.05,2.7]))
    self.state_hist = np.zeros((self.ns+1,self.x0.shape[0]))
    self.state = np.array([20.5, 0.8861, 0.1082, 0.0058, 21.5, 0.8861, 0.1082, 0.0058, 20.5, 0.1139, 0.7779, 0.1082,self.SP[0,0],self.SP[1,0],self.SP[2,0]])
    self.e_history = []
    self.info['state'] = self.state[:self.Nx]
    self.i = 0
    self.done = False
    rl_state = [self.state[i] for i in [0, 4, 8,0, 4, 8,12,13,14]]

    rl_state.append(self.F0[0,0])
    self.state_hist[self.i] = self.state
    self.F0_i = 0
    self.norm_state = (rl_state - self.observation_space_actual.low) / (self.observation_space_actual.high-self.observation_space_actual.low)
    return self.norm_state, {}
  def step(self, action):
    action  = (action + 1)/2
    
    if self.i % 5 == 0:
      self.action = action * (self.PID_space.high - self.PID_space.low) + self.PID_space.low
    try:
      self.state[:self.Nx] = self.integrate(self.action)
      noise_percentage = 0.01
      self.state[0] += np.random.uniform(-1,1)*noise_percentage
      self.state[4] += np.random.uniform(-1,1)*noise_percentage
      self.state[8] += np.random.uniform(-1,1)*noise_percentage     
    except:
      print('Integration Error')
      rew = -1e5
      self.done = True
    rew = self.reward(self.state[:self.Nx])
    self.info['PID_Action'] = self.action
    self.i += 1
    if self.i == self.ns and self.F0_i == 2:
      self.done = True
    if self.i == self.ns and self.test:
      self.done = True
   
    if self.i == self.ns and not self.test and not self.F0_i == 2:
      self.F0_i += 1
      self.state_hist = np.zeros((self.ns+1,self.x0.shape[0]))
      self.state = np.array([20.5, 0.8861, 0.1082, 0.0058, 21.5, 0.8861, 0.1082, 0.0058, 20.5, 0.1139, 0.7779, 0.1082,self.SP[0,0],self.SP[1,0],self.SP[2,0]])
      self.e_history = []
      self.i = 0
      rl_state = [self.state[i] for i in [0, 4, 8,0, 4, 8,12,13,14]]
      rl_state.append(self.F0[self.F0_i,0])
      self.state_hist[self.i] = self.state
      rew = 0

    if self.i > 0:
      self.state_hist[self.i] = self.state
      rl_state = np.array([self.state[0],self.state[4],self.state[8],self.state_hist[self.i-1][0],self.state_hist[self.i-1][4],self.state_hist[self.i-1][8],self.state[12],self.state[13],self.state[14]])
      if self.i < self.ns/3:
        rl_state = np.append(rl_state,self.F0[self.F0_i,0])
      else: 
        rl_state = np.append(rl_state,self.F0[self.F0_i,1])
    
    self.norm_state = (rl_state - self.observation_space_actual.low) / (self.observation_space_actual.high-self.observation_space_actual.low)
  
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
      uk[4] = self.F0[self.F0_i,0]
    else: 
      uk[4] = self.F0[self.F0_i,1]

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