import gymnasium as gym
from gymnasium import spaces 
from stable_baselines3 import SAC
from casadi import *
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# Import Environment
from RSR_Model_1602 import RSR

# Initialise Env
ns = 300
env = RSR(ns,test=False,plot=False)

def plot_simulation(s_DFO,s_SAC,a_DFO,a_SAC,c_DFO,c_SAC,ns):
    #Enable LaTeX for the plots
    plt.rcParams['text.usetex'] = 'True'#

    #Load Data
    DFO_a_data = np.array(a_DFO)
    DFO_c_data = np.array(c_DFO)
    SAC_a_data = np.array(a_SAC)
    SAC_c_data = np.array(c_SAC)
    SP = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]]).reshape(ns,1)
    SP_M = np.array([[21 for i in range(int(ns/2))] + [21 for i in range(int(ns/2))]]).reshape(ns,1)
    DFO_s_data = [np.array(s_DFO)[:,i] for i in [0,4,8,1,2,3,5,6,7,9,10,11]]
    SAC_s_data = [np.array(s_SAC)[:,i] for i in [0,4,8,1,2,3,5,6,7,9,10,11]]

    #Colours, Titles and Labels
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

    # DFO Holdup Plots
    for i in range(3):
        axs[0,0].plot(t,DFO_s_data[i], color=colors[i], label= labels[i])
        axs[0,0].set_ylabel('Vessel Holdup')
    #axs[0,0].step(t, SP, 'k--', where  = 'post',label='SP$_R$ \& SP$_B$')
    axs[0,0].set_title('DFO Holdups', y=1.2)
    axs[0,0].step(t, SP_M, 'k-.', where  = 'post',label='SP')
    axs[0,0].axvline(x = t[int((ns-1)/3)], alpha = 0.7, linestyle = 'dashed',label = '$F_0$ Disturbance') 
    axins = zoomed_inset_axes(axs[0,0], zoom=3, loc='upper right')  
    for i in range(3):
        axins.plot(t, DFO_s_data[i], color=colors[i])
    axins.grid(True)
    axins.axvline(x = t[int((ns-1)/3)], alpha = 0.7, linestyle = 'dashed') 
    axins.step(t, SP_M, 'k-.', where  = 'post')
    axins.set_xlim(6, 10)
    axins.set_ylim(20.9, 21.2)
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    mark_inset(axs[0,0], axins, loc1=2, loc2=4, fc="none", ec="0.5")

    axs[0,0].legend(loc='upper center', bbox_to_anchor=(1.2, 1.2),
          ncol=5,frameon=False)
    axs[0,0].set_ylim(19.5, 22.7)
    
    #SAC Holdup Plots
    for i in range(3):
        axs[0,1].plot(t,SAC_s_data[i], color=colors[i], label='SAC ' + labels[i])
        axs[0,1].set_ylabel('Vessel Holdup')
    axs[0,1].set_title('SAC Holdups',y = 1.2)
    #axs[0,0].step(t, SP, 'k--', where  = 'post',label='SP$_R$ \& SP$_B$')
    axs[0,1].step(t, SP_M, 'k-.', where  = 'post',label='SP')
    axs[0,1].axvline(x = t[int((ns-1)/3)], alpha = 0.7, linestyle = 'dashed',label = '$F_0$ Disturbance') 
    axins = zoomed_inset_axes(axs[0,1], zoom=3, loc='upper right')  
    for i in range(3):
        axins.plot(t, SAC_s_data[i], color=colors[i])
    axins.grid(True)
    axins.axvline(x = t[int((ns-1)/3)], alpha = 0.7, linestyle = 'dashed',label = '$F_0$ Disturbance') 
    axins.step(t, SP_M, 'k-.', where  = 'post')
    axins.set_xlim(6, 10)
    axins.set_ylim(20.9, 21.2)
    axins.xaxis.set_visible(False)
    axins.yaxis.set_visible(False)
    mark_inset(axs[0,1], axins, loc1=2, loc2=4, fc="none", ec="0.5")
    # axs[0,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
    #       ncol=3,frameon=False)
    
    axs[0,1].set_ylim(19.5, 22.7)
    #PID Action Plots
    for i in range(0,4):
        axs[1,0].step(t, DFO_a_data[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='DFO ' + PID_labels[i % len(PID_labels)])
        axs[1,0].step(t, SAC_a_data[:,i], where='post', color=colors_PID[i % len(colors_PID)], label='SAC ' + PID_labels[i % len(PID_labels)])
    axs[1,0].set_ylabel('$F_R$ PID Action')
    axs[1,0].legend(loc='upper center', bbox_to_anchor=(1.2, 1.35),
          ncol=4,frameon=False)
    for i in range(4,8):
        axs[1,1].step(t, DFO_a_data[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='DFO ' +PID_labels[i % len(PID_labels)])
        axs[1,1].step(t, SAC_a_data[:,i], where='post', color=colors_PID[i % len(colors_PID)], label='SAC ' +PID_labels[i % len(PID_labels)])
    axs[1,1].set_ylabel('$F_M$ PID Action')
    #axs[1,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)
    for i in range(8,12):
        axs[2,0].step(t, DFO_a_data[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='DFO' +PID_labels[i % len(PID_labels)])
        axs[2,0].step(t, SAC_a_data[:,i], where='post', color=colors_PID[i % len(colors_PID)], label='SAC' +PID_labels[i % len(PID_labels)])
    axs[2,0].set_ylabel('$B$ PID Action')
    #axs[2,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)
    
    for i in range(12,16):
        axs[2,1].step(t, DFO_a_data[:,i], where='post', linestyle='dashed', color=colors_PID[i % len(colors_PID)], label='DFO' +PID_labels[i % len(PID_labels)])
        axs[2,1].step(t, SAC_a_data[:,i], where='post', color=colors_PID[i % len(colors_PID)], label='SAC' +PID_labels[i % len(PID_labels)])
    axs[2,1].set_ylabel('$D$ PID Action')
    #axs[2,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          #ncol=4,frameon=False)

    for i in range(2):
        axs[3,0].step(t, DFO_c_data[:,i], where='post', linestyle='dashed', color=colors[i % len(colors)], label='DFO ' +control_labels[i])
        axs[3,0].step(t, SAC_c_data[:,i], where='post', color=colors[i % len(colors)], label='SAC ' +control_labels[i])
    axs[3,0].set_ylabel('Flowrate (h$^{-1}$)')
    axs[3,0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
          ncol=2,frameon=False)
   
 
    for i in range(2,4):
        axs[3,1].step(t, DFO_c_data[:,i], where='post', linestyle='dashed', color=colors[i % len(colors)], label='DFO ' +control_labels[i])
        axs[3,1].step(t, SAC_c_data[:,i], where='post', color=colors[i % len(colors)], label='SAC ' +control_labels[i])
    axs[3,1].set_ylabel('Flowrate (h$^{-1}$)')
    axs[3,1].legend(loc='upper center', bbox_to_anchor=(0.5, 1.35),
          ncol=2,frameon=False)
    
    plt.subplots_adjust(hspace = 0.6)
    plt.subplots_adjust(wspace = 0.3)
    plt.savefig('PG_DFO_Comparison_1602.pdf')
    plt.show()


def rollout(ns,policy):
    env = RSR(ns,test=False,plot=False)
    s,_ = env.reset()
    done = False
    states = []
    actions = []   
    rewards = []
    controls = []
    tot_rew = 0
    while not done:
        
        action = policy.predict(s,deterministic = True)[0]
        
        s, reward, done, _, control = env.step(action)
        #un normalise state
        states.append(control['state'])
        actions.append(control['PID_Action'])
        rewards.append(reward)
        tot_rew += reward
        controls.append(control['control_in'])

    return states, actions, tot_rew,controls
def rollout_DFO(ns,policy):
    env = RSR(ns,test=False,plot=False)
    s,_ = env.reset()
    done = False
    states = []
    actions = []
    rewards = []
    controls = []
    tot_reward = 0
    while not done:
        action = policy.predict(torch.tensor(s))[0].detach().numpy()
       
        s, reward, done, _,control = env.step(action)
        #un normalise state
        states.append(control['state'])
      
        actions.append(control['PID_Action'])
        rewards.append(reward)
        tot_reward += reward
        controls.append(control['control_in'])
    
    return states, actions, tot_reward,controls
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
  

# Rollout DFO 
best_policy_sd = torch.load('DFO_best_policy_1602.pth')
policy_plot = Net(n_fc1 = 256,n_fc2 = 256,activation = torch.nn.ReLU,n_layers = 1,deterministic = True) # Deterministic for plotting
policy_plot.load_state_dict(best_policy_sd)
s_DFO,a_DFO,r_DFO,c_DFO = rollout_DFO(ns,policy_plot)

# Rollout SAC
model = SAC.load('./logs/SAC_model_1502_7200_steps.zip')
env = RSR(ns,test = False,plot= False)
s_SAC,a_SAC,r_SAC,c_SAC = rollout(ns, model)

# Plot Comparison
print('SAC ISE:',np.round(r_SAC,2)*-1)
print('DFO ISE:',np.round(r_DFO,2)*-1)
plot_simulation(s_DFO,s_SAC,a_DFO,a_SAC,c_DFO,c_SAC,ns)