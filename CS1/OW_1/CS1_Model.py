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
    u = min(max(u,290),)


    return u
def PID_velocity(Ks,e,e_history,u_prev,ts,s_hist):
    # K gains
    dt = ts[1] - ts[0]
    KpCa = Ks[0]; KiCa = Ks[1] + 1e-6; KdCa = Ks[2]
    KpF = Ks[3]; KiF = Ks[4] + 1e-6; KdF = Ks[5] 
   
   
    
    Tc = u_prev[-1,0] + KpCa*(e[0] - e_history[-1,0]) + (KpCa/KiCa)*e[0]*dt - KpCa*KdCa*(e[0]-2*e_history[-1,0]+e_history[-2,0])/dt
    
    Tc = min(max(Tc,290),450)

    F = u_prev[-1,1] + KpF*(e[1] - e_history[-1,1]) + (KpF/KiF)*e[1]*dt - KpF*KdF*(e[1]-2*e_history[-1,1]+e_history[-2,1])/dt

    F = min(max(F,97),105)

    u = np.array([Tc,F])
    return u


def cstr_CS1(x,t,u,Tf,Caf,k0,UA):

    # ==  Inputs (2) == #
    Tc  = u[0] # Temperature of Cooling Jacket (K)
    Fin = u[1] # Volumetric Flowrate at inlet (m^3/sec) = 100

    # == States (5) == #
    Ca = x[0] # Concentration of A in CSTR (mol/m^3)
    Cb = x[1] # Concentration of B in CSTR (mol/m^3)
    Cc = x[2] # Concentration of C in CSTR (mol/m^3)
    T  = x[3] # Temperature in CSTR (K)
    V  = x[4] # Volume in CSTR (K)

    # == Process parameters == #
    Tf       = 350    # Feed Temperature (K)
    Caf      = 1      # Feed Concentration of A (mol/m^3)
    Fout     = 100    # Volumetric Flowrate at outlet (m^3/sec)
    #V       = 100    # Volume of CSTR (m^3)
    rho      = 1000   # Density of A-B Mixture (kg/m^3)
    Cp       = 0.239  # Heat Capacity of A-B-C Mixture (J/kg-K)
    UA       = 5e4    # U -Heat Transfer Coefficient (W/m^2-K) A -Area - (m^2)
    # Reaction A->B
    mdelH_AB  = 5e3    # Heat of Reaction for A->B (J/mol)
    EoverR_AB = 8750   # E -Activation Energy (J/mol), R -Constant = 8.31451 J/mol-K
    k0_AB     = 7.2e10 # Pre-exponential Factor for A->B (1/sec)#
    rA        = k0_AB*np.exp(-EoverR_AB/T)*Ca # reaction rate
    # Reaction B->C
    mdelH_BC  = 4e3      # Heat of Reaction for B->C (J/mol) => 5e4
    EoverR_BC = 10750    # E -Activation Energy (J/mol), R -Constant = 8.31451 J/mol-K !! 10
    k0_BC     = 8.2e10   # Pre-exponential Factor for A->B (1/sec)# !! 8
    rB        = k0_BC*np.exp(-EoverR_BC/T)*Cb # reaction rate !! **2
    # play with mdelH_BC, factor on Cb**2 and k0_BC, maybe even EoverR_BC

    # == Concentration Derivatives == #
    dCadt    = (Fin*Caf - Fout*Ca)/V - rA  # A Concentration Derivative
    dCbdt    = rA - rB - Fout*Cb/V         # B Concentration Derivative
    dCcdt    = rB      - Fout*Cc/V         # B Concentration Derivative
    dTdt     = Fin/V*(Tf - T) \
              + mdelH_AB/(rho*Cp)*rA \
              + mdelH_BC/(rho*Cp)*rB \
              + UA/V/rho/Cp*(Tc-T)   # Calculate temperature derivative
    dVdt     = Fin - Fout

    # == Return xdot == #
    xdot    = np.zeros(5)
    xdot[0] = dCadt
    xdot[1] = dCbdt
    xdot[2] = dCcdt
    xdot[3] = dTdt
    xdot[4] = dVdt
    return xdot


# Create a gym environment
class reactor_class(gym.Env):
  def __init__(self,ns,test = False, DR = False,robust_test = False,PID_pos = False, PID_vel = False,DS = False):
    self.DS = DS
    self.PID_pos = PID_pos
    self.PID_vel = PID_vel
    self.x_norm = np.array(([-10,0,0.01],[10,20,10]))
    self.ns = ns 
    self.test = test
    self.DR = DR
    self.robust_test = robust_test
    Ca_des1 = [0.61 for i in range(int(ns/2))] + [0.76 for i in range(int(ns/3))] + [0.87 for i in range(int(ns/2))]
    Ca_des2 = [0.6 for i in range(int(ns/2))] + [0.74 for i in range(int(ns/3))] +  [0.86 for i in range(int(ns/2))]
    Ca_des3 = [0.59 for i in range(int(ns/2))] + [0.75 for i in range(int(ns/3))] +  [0.85 for i in range(int(ns/2))]
    

    
    if self.test:
      #Ca_des1 = [0.05 for i in range(int(ns/3))] + [0.15 for i in range(int(ns/3))] + [0.25 for i in range(int(ns/3))]
      Ca_des1 = [0.3 for i in range(int(ns/3))] + [0.45 for i in range(int(ns/3))] + [0.6 for i in range(int(ns/3))]
      V_des = [100 for i in range(int(ns))]
    
      
    
    self.observation_space = spaces.Box(low = np.array([.70, 310,90,.70, 310,90,0.7]),high= np.array([1,340,200,1,340,200,1]))
    self.action_space = spaces.Box(low = np.array([-1]*3),high= np.array([1]*3))

    
    self.SP = np.array(([[Ca_des1],[Ca_des2],[Ca_des3]],[[V_des]]),dtype = object)

    self.Ca_ss = 0.80
    self.T_ss  = 327
    self.V_ss = 100
    self.x0    = np.empty(2)
    self.x0[0] = self.Ca_ss
    self.x0[1] = self.T_ss
  
    self.u_ss = 300.0 # Steady State Initial Condition

    self.Caf  = 1     # Feed Concentration (mol/m^3)

    # Time Interval (min)
    self.t = np.linspace(0,100,ns)

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
    Ca_des = self.SP[0][self.SP_i][0][0]
    V_des = self.SP[1][0][0][0]
    self.state = np.array([self.Ca_ss,0,0,self.T_ss,self.V_ss,self.Ca_ss,0,0,self.T_ss,self.V_ss,Ca_des,V_des])
    self.done = False
    if not self.test:
      self.disturb = False
    self.u_history = []
    self.e_history = []
    self.s_history = []
    self.ts = [self.t[self.i],self.t[self.i+1]]
    self.RL_state = [self.state[i] for i in [1,3,4,6,8,9,10]]
    self.state_norm = (self.RL_state -self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    return self.state_norm,{}

  def step(self, action_policy):
    if self.DS:
       self.u_DS = action_policy
       self.info = {}
    if self.i % 5 == 0:
      self.action = copy.deepcopy(action_policy)
    Ca_des = self.SP[0][self.SP_i][0][self.i]
    V_des = self.SP[1][0][0][self.i]
    
    self.state,rew = self.reactor(self.state,self.action,Ca_des,V_des)
    self.i += 1
    if self.i == self.ns:
        if self.test:
          self.done = True
        elif self.SP_i < 2:
          self.SP_i += 1
          self.i = 0
          Ca_des = self.SP[0][self.SP_i][0][0]
          V_des = self.SP[1][0][0][0]
          self.state = np.array([self.Ca_ss,0,0,self.T_ss,self.V_ss,self.Ca_ss,0,0,self.T_ss,self.V_ss,Ca_des,V_des])
          self.u_history = []
          self.e_history = []
        else:
          self.done = True

    self.RL_state = [self.state[i] for i in [1,3,4,6,8,9,10]]
    self.state_norm = (self.RL_state -self.observation_space.low)/(self.observation_space.high - self.observation_space.low)
    return self.state_norm,rew,self.done,False,self.info

  def reactor(self,state,action,Ca_des,V_des):
    if not self.DR or not self.robust_test:
      k0 = 7.2e10 #1/sec
      UA = 7e5 # W/K
    if self.robust_test or self.DR:
      k0 = self.k0
      UA = self.UA
   
    # Steady State Initial Conditions for the States

    Ca = state[0]
    Cb = state[1]
    Cc = state[2]
    T  = state[3]
    V  = state[4]

    x_sp    = np.array([Ca_des,V_des])
    e = np.zeros(2)
    e[0]  = x_sp[0] - state[1]
    e[1]  = x_sp[1] - state[4]
    
    
    Ks = copy.deepcopy(action) #Ca, T, u, Ca setpoint and T setpoint
    
    #Adjust bounds from relu
    if not self.DS:
      
      Ks_norm = ((Ks + 1) / 2) * (self.x_norm[1] - self.x_norm[0]) + self.x_norm[0]
   
      # Ks[3] = (Ks[3])*13 + 290
      self.info = {'Ks':Ks_norm}

      if self.PID_vel:
        if self.i < 2:
          u = np.array([302,100])
        else:
          u =  PID_velocity(Ks_norm,e,np.array(self.e_history),self.u_history,self.ts,np.array(self.s_history))
    if self.DS:
      u = self.u_DS
    # simulate system

    y       = odeint(cstr_CS1,state[0:5],self.ts,args=(u,self.Tf,self.Caf,k0,UA))

    # add process disturbance

    Ca_plus = y[-1][0] #+ np.random.uniform(low=-0.00075,high=0.00075)
    Cb_plus = y[-1][1] #+ np.random.uniform(low=-0.00075,high=0.00075)
    Cc_plus = y[-1][2] #+ np.random.uniform(low=-0.00075,high=0.00075)
    T_plus  = y[-1][3] #+ np.random.uniform(low=-.025,high=0.025)
    V_plus = y[-1][4] #+ np.random.uniform(low=-.025,high=0.025)
    # collect data
    state_plus = np.zeros(11)
    state_plus[0]   = Ca_plus
    state_plus[1]   = Cb_plus
    state_plus[2]   = Cc_plus
    state_plus[3]   = T_plus
    state_plus[4] = V_plus
    state_plus[5]   = Ca
    state_plus[6]   = Cb
    state_plus[7]   = Cc
    state_plus[8]   = T
    state_plus[9]   = V
    state_plus[10]   = Ca_des
    # compute tracking error
    self.e_history.append((e))
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
  
    r_x = ((e[0])**2) * 1e2 #+ u_mag + u_cha
    r_x += ((e[1])**2) 
    return state_plus, -r_x