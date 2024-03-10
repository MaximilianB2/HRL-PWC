import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import odeint
import copy


def PID(Ks, x, x_setpoint, e_history):

    # K gains
    KpCa = Ks[0]; KiCa = Ks[1]; KdCa = Ks[2]
    Kb = Ks[3]
    # setpoint error

    e = x_setpoint - x
    # if Ks[0] == 0 and Ks[1] == 0 and Ks[2] == 0:
    #   e = x - x_setpoint
    # control action

    
    u = KpCa*e[0] + (KpCa/KiCa)*sum(e_history[:,0]) + KpCa*KdCa*(e[0]-e_history[-1,0])
    u += Kb
    u = min(max(u,290),303)


    return u
def PID_velocity(Ks,x,x_setpoint,e_history,u_prev,ts,s_hist):
    # K gains
    dt = ts[1] - ts[0]
    KpCa = Ks[0]; KiCa = Ks[1]; KdCa = Ks[2]
    e = x_setpoint - x
   
    
    u = u_prev[-1] + KpCa*(e[0] - e_history[-1,0]) + (KpCa/KiCa)*e[0]*dt - KpCa*KdCa*(e[0]-2*e_history[-1,0]+e_history[-2,0])/dt
    
    u = min(max(u,290),303)
    return u


def cstr_CS1(x,t,u,Tf,Caf,k0,UA):

    # ==  Inputs (3) == #
    Tc = u # Temperature of cooling jacket (K)
    # Tf = Feed Temperature (K)
    # Caf = Feed Concentration (mol/m^3)

    # == States == #
    Ca = x[0] # Concentration of A in CSTR (mol/m^3)
    T  = x[1] # Temperature in CSTR (K)

    # == Process parameters == #
    q      = 100    # Volumetric Flowrate (m^3/sec)
    V      = 100    # Volume of CSTR (m^3)
    rho    = 1000   # Density of A-B Mixture (kg/m^3)
    Cp     = 0.239  # Heat capacity of A-B Mixture (J/kg-K)
    mdelH  = 5e4    # Heat of reaction for A->B (J/mol)
    EoverR = 8750   # E -Activation energy (J/mol), R -Constant = 8.31451 J/mol-1
    rA     = k0*np.exp(-EoverR/T)*Ca # reaction rate
    dCadt  = q/V*(Caf - Ca) - rA     # Calculate concentratioing n derivative
    dTdt   = q/V*(Tf - T) \
              + mdelH/(rho*Cp)*rA \
              + UA/V/rho/Cp*(Tc-T)   # Calculate temperature derivative

    # == Return xdot == #
    xdot    = np.zeros(2)
    xdot[0] = dCadt
    xdot[1] = dTdt
    
    return xdot


# Create a gym environment
class reactor_class(gym.Env):
  def __init__(self,ns,test = False, DR = False,robust_test = False,PID_pos = False, PID_vel = False,DS = False):
    self.DS = DS
    self.PID_pos = PID_pos
    self.PID_vel = PID_vel
  
    self.ns = ns 
    self.test = test
    self.DR = DR
    self.robust_test = robust_test
    Ca_des1 = [0.95 for i in range(int(ns/2))] + [0.85 for i in range(int(ns/2))]
    T_des1  = [330 for i in range(int(ns/2))] + [320 for i in range(int(ns/2))]

    Ca_des2 = [0.9 for i in range(int(ns/2))] + [0.95 for i in range(int(ns/2))]
    T_des2  = [340 for i in range(int(ns/2))] + [320 for i in range(int(ns/2))]

    Ca_des3 = [0.95 for i in range(int(ns/2))] + [0.8 for i in range(int(ns/2))]
    T_des3  = [320 for i in range(int(ns/2))] + [330 for i in range(int(ns/2))]

    Ca_des4 = [0.9 for i in range(int(ns/2))] + [0.85 for i in range(int(ns/2))]
    T_des4  = [320 for i in range(int(ns/2))] + [340 for i in range(int(ns/2))]
    if self.test:
      
      Ca_des1 = [0.95 for i in range(int(ns/3))] + [0.9 for i in range(int(ns/3))] + [0.85 for i in range(int(ns/3))]     
      T_des1  = [325 for i in range(int(2*ns/5))] + [320 for i in range(int(ns/5))] + [327 for i in range(int(2*ns/5))]
       
      
    
    self.observation_space = spaces.Box(low = np.array([.70, 315,.70, 315,0.75, 320]),high= np.array([0.95,340,0.95,340,0.95,340]))
    self.action_space = spaces.Box(low = np.array([-1]*4),high= np.array([1]*4))


    
    self.SP = np.array(([Ca_des1,T_des1],[Ca_des2,T_des2],[Ca_des3,T_des3],[Ca_des4,T_des4]),dtype = object)

    self.Ca_ss = 0.87725294608097
    self.T_ss  = 324.475443431599
    self.x0    = np.empty(2)
    self.x0[0] = self.Ca_ss
    self.x0[1] = self.T_ss
  
    self.u_ss = 300.0 # Steady State Initial Condition

    self.Caf  = 1     # Feed Concentration (mol/m^3)

    # Time Interval (min)
    self.t = np.linspace(0,50,ns)

    # Store results for plotting
    self.Ca = np.ones(len(self.t)) * self.Ca_ss
    self.T  = np.ones(len(self.t)) * self.T_ss
    self.u  = np.ones(len(self.t)) * self.u_ss
    self.Ks_list = np.zeros((7,len(self.t)))
    self.Tf = 350 # Feed Temperature (K)

    
  def UA_dist(self):
    sample = np.random.normal(0, 1.35e4)
    return 5e4 - abs(sample)
  

  def UA_dist_test(self):
     sample = np.random.uniform(0, 1.35e4)
     return 5e4 - abs(sample)
  

  def k0_dist_test(self):
    sample = np.random.uniform(0, 3e10)
    return 7.2e10 - abs(sample)
  

  def k0_dist(self):
    sample = np.random.normal(0, 3e10)
    return 7.2e10 - abs(sample)
  

  def reset(self, seed = None):
    self.i = 0
    self.SP_i = 0
    if self.DR:
      self.UA = self.UA_dist()
      self.k0 = self.k0_dist()
    elif self.robust_test:
      self.UA = self.UA_dist_test()
      self.k0 = self.k0_dist_test()
    Ca_des = self.SP[self.SP_i,0][self.i]
    T_des = self.SP[self.SP_i,1][self.i] 
    self.state = np.array([self.Ca_ss,self.T_ss,self.Ca_ss,self.T_ss,Ca_des,T_des])
    self.done = False
    if not self.test:
      self.disturb = False
    self.u_history = []
    self.e_history = []
    self.s_history = []
    self.ts = [self.t[self.i],self.t[self.i+1]]
    self.state_norm = (self.state -self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    return self.state_norm,{}

  def step(self, action_policy):
    if self.DS:
       self.u_DS = action_policy
    if self.i % 5 == 0:
       self.action = copy.deepcopy(action_policy)
    Ca_des = self.SP[self.SP_i,0][self.i]
    T_des = self.SP[self.SP_i,1][self.i]  
    
    self.state,rew = self.reactor(self.state,self.action,Ca_des,T_des)
    self.i += 1
    if self.i == self.ns:
        if self.SP_i < 3:
          self.SP_i += 1
          self.i = 0
          self.state = np.array([self.Ca_ss,self.T_ss,self.Ca_ss,self.T_ss,Ca_des,T_des])
          self.u_history = []
          self.e_history = []
        else:
          self.done = True
          
    
    self.state_norm = (self.state -self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    return self.state_norm,rew,self.done,False,self.info

  def reactor(self,state,action,Ca_des,T_des):
    if not self.DR or not self.robust_test:
      k0 = 7.2e10 #1/sec
      UA = 5e4 # W/K
    if self.robust_test or self.DR:
      k0 = self.k0
      UA = self.UA
   
    # Steady State Initial Conditions for the States

    Ca = state[0]
    T  = state[1]
  
    x_sp    = np.array([Ca_des,T_des])
    
    
    Ks = copy.deepcopy(action) #Ca, T, u, Ca setpoint and T setpoint
    
    #Adjust bounds from relu
    if not self.DS:
      x_norm = np.array(([-200,0,0.01],[0,20,10]))
      Ks_norm = ((Ks + 1) / 2) * (x_norm[1] - x_norm[0]) + x_norm[0]
   
      # Ks[3] = (Ks[3])*13 + 290
      self.info = {'Ks':Ks_norm}

      if self.PID_vel:
        if self.i < 2:
          u = 295
        else:
          u =  PID_velocity(Ks_norm,state[0:2], x_sp, np.array(self.e_history),self.u_history,self.ts,np.array(self.s_history))
    if self.DS:
      u = self.u_DS
    # simulate system

    y       = odeint(cstr_CS1,state[0:2],self.ts,args=(u,self.Tf,self.Caf,k0,UA))

    # add process disturbance
    Ca_plus = y[-1][0] + np.random.uniform(low=-0.001,high=0.001)
    T_plus  = y[-1][1] + np.random.uniform(low=-.5,high=0.5)
    # collect data
    state_plus = np.zeros(6)
    state_plus[0]   = Ca_plus
    state_plus[1]   = T_plus
    state_plus[2]   = Ca
    state_plus[3]   = T
    state_plus[4]   = Ca_des
    state_plus[5]   = T_des
    # compute tracking error
    e  = x_sp-state[0:2]
    self.e_history.append((x_sp-state[0:2]))
    #Penalise control action magnitude
    u_mag = np.abs(np.array(u-290))/10 #295 is lower bound of jacket temperature
    u_mag = u_mag/10
    # penalise change in control action
    if self.i == 0:
      u_cha = 0
    else:
      u_cha = np.abs(u-self.u_history[-1])/100
    
   
    self.u_history.append(u)
    self.s_history.append(state[0:2])
    #Compute reward (squared distance + scaled)
  
    r_x = ((e[0])**2) * 1e2 + u_mag + u_cha

    return state_plus, -r_x