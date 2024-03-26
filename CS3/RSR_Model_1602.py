import gymnasium as gym
from gymnasium import spaces 
from casadi import *
import numpy as np
import copy
class RSR(gym.Env):
  def __init__(self,ns,test, plot = False, DS = False, DS_uk = np.zeros(2)):
    self.D = DS_uk[1]
    self.Tc = DS_uk[0]
    self.DS = DS
    self.i = 0
    self.ns = ns 
    self.T  = 20000
    self.dt = self.T/self.ns
    self.Nx = 13
    self.plot = plot  
    self.test = test
    self.info = {'control_in':0,'PID_Action':0,'state':0}

    # Time Interval (min)
    self.t = np.linspace(0,self.T,self.ns)

    # Casadi Model
    self.sym_x = self.gen_casadi_variable(13, "x")
    self.sym_u = self.gen_casadi_variable(6, "u")    
    large_scale_ode = self.large_scale_ode
    self.casadi_sym_model = self.casadify(large_scale_ode, self.sym_x, self.sym_u)
    self.casadi_model_func = self.gen_casadi_function([self.sym_x, self.sym_u],[self.casadi_sym_model],
                                                    "model_func", ["x","u"], ["model_rhs"])
    # Observation Space
    self.observation_space = spaces.Box(low = 0,high=1, shape = (10,))
    self.observation_space_actual = spaces.Box(low =np.array([18,18,18,18,18,18,18,18,18,0,290]) , high = np.array([35,35,35,35,35,35,35,35,35,1,350]))
    
    # Action Space
    self.action_space = spaces.Box(low = -1, high = 1, shape = (15,))
    self.action_space_unnorm = spaces.Box(low = np.array([8,8,0.67,8,200]), high = np.array([47,47,3,45,300]))
    self.PID_space = spaces.Box(low = np.array([1,0,0,  1,2,0,  1,1,0,  -20,0,0, -10,0,0]),
                                high = np.array([15,15,1,  15,5,1,  15,15,1,  20,15,1 ,10,20,1 ]))
    SPtest = [30, 30, 30]
    x_b_sp = [0.94,0.94, 0.94]
    self.SP_test = np.array([[SPtest[0] for i in range(int(ns/2))] + [35 for i in range(int(ns/2))],[SPtest[0] for i in range(int(ns/2))] + [25 for i in range(int(ns/2))],[SPtest[0] for i in range(int(ns/2))] + [25 for i in range(int(ns/2))],[x_b_sp[0] for i in range(int(ns/3))] + [x_b_sp[1] for i in range(int(ns/3))] +  [x_b_sp[2] for i in range(int(ns/3))]])
    
    SP1 = [21.5,20.7,19.3]
    SP2 = [21.7,20.5,19.7]
    SP3 = [21.3,20.3,19.5]
    self.SP1 = np.array([[SP1[0] for i in range(int(ns/3))] + [SP1[1] for i in range(int(ns/3))]+[SP1[2] for i in range(int(ns/3))],[SP1[0] for i in range(int(ns/3))] + [SP1[1] for i in range(int(ns/3))]+[SP1[2] for i in range(int(ns/3))],[SP1[0] for i in range(int(ns/3))] + [SP1[1] for i in range(int(ns/3))]+[SP1[2] for i in range(int(ns/3))]])
    self.SP2 = np.array([[SP2[0] for i in range(int(ns/3))] + [SP2[1] for i in range(int(ns/3))]+[SP2[2] for i in range(int(ns/3))],[SP2[0] for i in range(int(ns/3))] + [SP2[1] for i in range(int(ns/3))]+[SP2[2] for i in range(int(ns/3))],[SP2[0] for i in range(int(ns/3))] + [SP2[1] for i in range(int(ns/3))]+[SP2[2] for i in range(int(ns/3))]])
    self.SP3 =  np.array([[SP3[0] for i in range(int(ns/3))] + [SP3[1] for i in range(int(ns/3))]+[SP3[2] for i in range(int(ns/3))],[SP3[0] for i in range(int(ns/3))] + [SP3[1] for i in range(int(ns/3))]+[SP3[2] for i in range(int(ns/3))],[SP3[0] for i in range(int(ns/3))] + [SP3[1] for i in range(int(ns/3))]+[SP3[2] for i in range(int(ns/3))]])
    if self.test:
      self.x0 = copy.deepcopy(np.array([21.25, 0.8861, 0.1082, 0.0058, 21.25, 0.8861, 0.1082, 0.0058, 21.25, 0.1139, 0.9, 0.1082,self.SP_test[0,0],self.SP_test[1,0],self.SP_test[2,0]]))
      self.SP  = np.array([self.SP_test])
    else:
      self.x0 = copy.deepcopy(np.array([21.5, 0.8861, 0.1082, 0.0058, 21.5, 0.8861, 0.1082, 0.0058, 21.5, 0.1139, 0.9, 0.1082,self.SP1[0,0],self.SP1[1,0],self.SP1[2,0]]))
      self.SP = np.array((self.SP1,self.SP2,self.SP3))
    self.e_history = []
    self.u_history = []
    
  def reset(self, seed = None):
    self.state_hist = np.zeros((self.ns+1,17))
    if self.test:
      self.state = np.array([30, 0.8861, 0.1082, 0.0058, 30, 0.8861, 0.1082, 0.0058, 30, 0.1139, 0.92, 0.1082,380,self.SP[0,0,0],self.SP[0,1,0],self.SP[0,2,0],self.SP[0,3,0]])
    else:
      self.state = np.array([30, 0.8861, 0.1082, 0.0058, 30, 0.8861, 0.1082, 0.0058, 30, 0.1139, 0.92, 0.1082,380,self.SP[0,0,0],self.SP[0,1,0],self.SP[0,2,0],self.SP[0,3,0],350])
    self.e_history = []
    self.u_history = []
    self.info['state'] = self.state[:self.Nx]
    self.i = 0
    self.done = False
    rl_state = [self.state[i] for i in [0, 4, 8,0, 4, 8,12,13,14,15,16]]


    self.state_hist[self.i] = self.state
    self.SP_i = 0
    self.norm_state = (rl_state - self.observation_space_actual.low) / (self.observation_space_actual.high-self.observation_space_actual.low)
    return self.norm_state, {}
  def step(self, action):
    action  = (action + 1)/2
    
    if self.i % 5 == 0:
      self.action = action * (self.PID_space.high - self.PID_space.low) + self.PID_space.low
    try:
      self.state[:self.Nx] = self.integrate(self.action)
      # noise_percentage = 0.05
      # self.state[0] += np.random.uniform(-1,1)*noise_percentage
      # self.state[4] += np.random.uniform(-1,1)*noise_percentage
      # self.state[8] += np.random.uniform(-1,1)*noise_percentage 
      rew = self.reward(self.state[:self.Nx])   
    except:
      print('Integration Error')
      rew = -1e5
      self.done = True
    
    self.info['PID_Action'] = self.action
    self.i += 1
    if self.i == self.ns and self.SP_i == 2:
      self.done = True
    if self.i == self.ns and self.test:
      self.done = True
   
    if self.i == self.ns and not self.test and not self.SP_i == 2:
      self.SP_i += 1
      self.state_hist = np.zeros((self.ns+1,self.x0.shape[0]))
      self.state = np.array([20.5, 0.8861, 0.1082, 0.0058, 21.5, 0.8861, 0.1082, 0.0058, 20.5, 0.1139, 0.7779, 0.1082,380,self.SP[self.SP_i,0,0],self.SP[self.SP_i,1,0],self.SP[self.SP_i,2,0],self.SP[self.SP_i,3,0]])
      self.e_history = []
      self.u_history = []
      self.i = 0
      rl_state = [self.state[i] for i in [0, 4, 8,0, 4, 8,12,13,14,15,16]]
      self.state_hist[self.i] = self.state
      rew = 0

    if self.i > 0:
      self.state_hist[self.i] = self.state
      rl_state = np.array([self.state[0],self.state[4],self.state[8],self.state_hist[self.i-1][0],self.state_hist[self.i-1][4],self.state_hist[self.i-1][8],self.state[12],self.state[13],self.state[14],self.state[15],self.state[16]])
    
    self.norm_state = (rl_state - self.observation_space_actual.low) / (self.observation_space_actual.high-self.observation_space_actual.low)
  
    return self.norm_state,rew,self.done,False,self.info

  
  def reward(self, state):
    if self.i == 0:
      u_mag = 0
      u_cha = 0 
    else:
      
      u_hist_norm = (np.array(self.u_history)[:,:5] - self.action_space_unnorm.low)/(self.action_space_unnorm.high - self.action_space_unnorm.low)
      u_mag = np.sum(np.abs(u_hist_norm[-1][:4]))
      u_cha = np.sum(np.abs(u_hist_norm[-1] - u_hist_norm[-2]))*50

    
    state = [state[i] for i in [0, 4, 8, 10]]
    obs_low = np.array([self.observation_space_actual.low[i] for i in [0, 4, 8]])
    obs_high = np.array([self.observation_space_actual.high[i] for i in [0, 4, 8]])
   
    # SP_norm = (self.SP[self.SP_i,:,self.i] - obs_low)/(obs_high-obs_low)
    # state_norm = (state - obs_low)/(obs_high-obs_low)
    ISE = np.sum((self.SP[self.SP_i,3,self.i] - state[3])**2)*10
    
    # print('ISE', ISE,'u_mag',u_mag,'u_cha',u_cha)
    r = ISE*100 #+ u_cha
    
    return -r
     
  def integrate(self, PID_gains):
    
    state = self.state[:self.Nx]
    Holdups = [state[i] for i in [0, 4, 8,10]]
    self.e = np.array([Holdups])-self.SP[self.SP_i,:,self.i] 
   
    uk = np.zeros(6)

    uk[0] = 20#self.PID_F_R(PID_gains[0:3])
    uk[1] = 20#self.PID_F_M(PID_gains[3:6])
    uk[2] = 1.5#self.PID_B(PID_gains[6:9])
    uk[3] = 18.5#self.PID_D(PID_gains[9:12])
    uk[4] = self.PID_T(PID_gains[12:15])
    uk[5] = 1.5
    

    if self.DS:
      uk = np.array([20,20,1.5,self.D,self.Tc,1.5])

    self.u_history.append(uk)
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
    k_i = PID_gains[1] + 1e-5    
    k_d = PID_gains[2]

    e_history = np.array(self.e_history)
    u_history = np.array(self.u_history)
    e = self.e[0]
    if self.i < 2:
      e_history = np.zeros((1,3))
      u = (self.action_space_unnorm.high[0] - self.action_space_unnorm.low[0])/2
    else:
      u = u_history[-1,0] + k_p*(e[0] - e_history[-1,0]) + (k_p/k_i)*e[0]*self.dt - k_p*k_d*(e[0]-2*e_history[-1,0]+e_history[-2,0])/self.dt
 
  
    u = np.clip(u, self.action_space_unnorm.low[0], self.action_space_unnorm.high[0])

    return u
  
  def PID_F_M(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1] + 1e-5    
    k_d = PID_gains[2]
    
    e_history = np.array(self.e_history)
    u_history = np.array(self.u_history)
    e = self.e[0]
    
    if self.i < 2:
      e_history = np.zeros((1,3))
      u = (self.action_space_unnorm.high[1] - self.action_space_unnorm.low[1])/2
    else:
      u = u_history[-1,1] + k_p*(e[1] - e_history[-1,1]) + (k_p/k_i)*e[1]*self.dt - k_p*k_d*(e[1]-2*e_history[-1,1]+e_history[-2,1])/self.dt
    
    # u = np.clip(u, self.action_space_unnorm.low[1], self.action_space_unnorm.high[1])
    u = np.clip(u, 19.5, 21)
  
    return u
     
  def PID_B(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1] + 1e-5    
    k_d = PID_gains[2]

    e_history = np.array(self.e_history)
    u_history = np.array(self.u_history)

    e = self.e[0]
    if self.i < 2:
      e_history = np.zeros((1,3))
      u = (self.action_space_unnorm.high[2] - self.action_space_unnorm.low[2])/2
    else:
       u = u_history[-1,2] + k_p*(e[2] - e_history[-1,2]) + (k_p/k_i)*e[2]*self.dt - k_p*k_d*(e[2]-2*e_history[-1,2]+e_history[-2,2])/self.dt
    u = np.clip(u, self.action_space_unnorm.low[2], self.action_space_unnorm.high[2])
    
    return u
    
  def PID_D(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1] + 1e-5    
    k_d = PID_gains[2]
    
    e_history = np.array(self.e_history)
    u_history = np.array(self.u_history)
    e = self.e[0]
    if self.i < 2:
      e_history = np.zeros((1,3))
      u = (self.action_space_unnorm.high[3] - self.action_space_unnorm.low[3])/2
    else:
      u = u_history[-1,3] + k_p*(e[3] - e_history[-1,3]) + (k_p/k_i)*e[3]*self.dt - k_p*k_d*(e[3]-2*e_history[-1,3]+e_history[-2,3])/self.dt

    #u = np.clip(u, self.action_space_unnorm.low[3], self.action_space_unnorm.high[3])
    u = np.clip(u, 18.2, 19)
    return u
  def PID_T(self, PID_gains):
    k_p = PID_gains[0]
    k_i = PID_gains[1] + 1e-5
    k_d = PID_gains[2]
    e_history = np.array(self.e_history)
    u_history = np.array(self.u_history)
    e = self.e[0]
    if self.i < 2:
      e_history = np.zeros((1,3))
      u = 290
    else:
      u = u_history[-1,4] + k_p*(e[3] - e_history[-1,3]) + (k_p/k_i)*e[3]*self.dt - k_p*k_d*(e[3]-2*e_history[-1,3]+e_history[-2,3])/self.dt
    u = np.clip(u,275,305)
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
        x = SX.sym("x", 13)
        u = SX.sym("u", 6)
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
      k_1 = 7.2e9 #Rate constant
      k_2 = 5e8 #Rate constant
      A_R = 10 #Vessel area
      A_M = 10 #Vessel area
      A_B = 5 #Vessel area
      x1_O = 1.00
      UA = 9e5
      Cp     = 0.239  # Heat capacity of A-B Mixture (J/kg-K)
      mdelH1  = 6e4    # Heat of reaction for A->B (J/mol)
      mdelH2  = 6e4 
      EoverR = 8750   # E -Activation energy (J/mol), R -Constant = 8.31451 J/mol-1
      Tf = 350
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

      H_R, x1_R, x2_R, x3_R, H_M, x1_M, x2_M, x3_M, H_B, x1_B, x2_B, x3_B,T = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11],x[12]

      ##Inputs##
      #F_O - Reactor input flowrate
      #F_R - Reactor outlet flowrate
      #F_M - Storage tank outlet flowrate
      #B - Bottoms flowrate
      #D - Distillate flowrate  
    
      F_R, F_M, B, D, Tc, F_O,=  u[0], u[1], u[2], u[3],u[4],u[5]

      #Calculate distillate composition (only component 1 and 2 are volatile and component 3 is not
      #while component 1 is 90 times more volatile than component 2)

      x1_D = ((x1_B * alpha_1) / (1 - x1_B + x1_B * alpha_1))
      x2_D = 1 - x1_D
    
      r_1     = k_1*np.exp(-EoverR/T)*x1_R*rho*A_R # reaction rate
      r_2     = k_2*np.exp(-EoverR/T)*x2_R*rho*A_R # reaction rate
      dxdt = [
          (1/(rho*A_R)) * (F_O + D - F_R),
          ((F_O*(x1_O - x1_R) + D*(x1_D - x1_R))/(rho*A_R*H_R)) - r_1,
          ((-F_O * x2_R + D * (x2_D - x2_R))/(rho*A_R*H_R)) + r_1 - r_2,
          ((-x3_R*(F_O + D))/(rho*A_R*H_R)) + r_2,
          (1/(rho*A_M)) * (F_R - F_M),
          ((F_R)/(rho*A_M*H_M))*(x1_R - x1_M),
          ((F_R)/(rho*A_M*H_M))*(x2_R - x2_M),
          ((F_R)/(rho*A_M*H_M))*(x3_R - x3_M),
          (1/(rho*A_B))*(F_M - B - D),
          (1/(rho*A_B*H_B))*(F_M*(x1_M-x1_B) - D*(x1_D - x1_B)),
          (1/(rho*A_B*H_B))*(F_M*(x2_M-x2_B) - D*(x2_D - x2_B)),
          (1/(rho*A_B*H_B))*(F_M*(x3_M-x3_B) + D*(x3_B)),
          (F_O/rho*A_R*(Tf - T) + mdelH1/(rho*Cp)*r_1 + mdelH2/(rho*Cp)*r_2 + UA/rho*A_R/rho/Cp*(Tc-T))
          ]

      return dxdt