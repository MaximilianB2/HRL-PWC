# Hierarchical Reinforcement Learning for Plant-wide Control
Written by Max Bloor and supervised by Dr Antonio del Rio Chanona and Niki Kotecha

## Abstract
 Proportional-Integral-Derivative (PID) controllers provide a simple and effective control solution used widely across the chemical industry. Their parameters require tuning with consideration of the entire system's dynamics to ensure adequate control. Traditional methods rely on system models or rules to tune these parameters. In this paper, a hierarchical reinforcement learning algorithm designed for plant-wide control is presented, consisting of a high-level artificial neural network and low-level PID controllers using an evolutionary algorithm for policy optimisation. Three case studies including a reactor, reactor-separator and reactor-separator-recycle systems are used to test the algorithm's performance at setpoint tracking, noise and disturbance rejection. The hierarchical reinforcement learning algorithm is compared to derivative-free optimisation, multiloop relay tuning, and a nonlinear model predictive controller (NMPC) for each case study. Across all case studies, the hierarchical reinforcement learning algorithm has a lower integral square error than both PID tuning methodologies. However, the NMPC showed that manipulation of other units in the system could result in enhanced setpoint tracking performance that PID-based methods could not replicate. The robustness of all controllers is investigated with a parametric mismatch analysis, which replicates the fouling of the reactor cooling jacket and the degradation of the reactor catalyst. This showed that due to the hierarchical reinforcement learning algorithm's lack of dependence on an accurate model, it can outperform the NMPC when there is a plant-model mismatch. 

## Run Case Studies

Case study code can be found in ``CS1.ipynb``, ``CS2.ipynb`` and ``CS3.ipynb``

## Case Study Results
Below is the simulation and parametric mismatch analysis of all case studies. In the parametric mismatch, the catalyst activity $k_0$ and reactor jacket heat transfer coefficient $UA$ were varied.
### Case Study 1 
In the first case study, a multiloop single continuously stirred tank reactor (CSTR) system is investigated. The objective is to track the setpoint for the output concentration of A ($C_A$) and the reactor temperature ($T$) by manipulating the jacket temperature ($T_J$). There is also a penalty for control inputs to promote smooth control. The reactor is assumed to be isothermal and perfectly mixed, and the reaction is irreversible and exothermic. This model is adapted from xxx. The simulation of this case study with the RL-PID, multiloop relay tuning, DFO-PID and NMPC control methods is shown below.

<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS1%20(1)-1.png" width="100%">
</P>

The parametric mismatch analysis is shown below.

<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS1_ParaMis_UA-1.png" width="50%">
</P>

<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS1_ParaMis_k0-1.png" width="50%">
</P>


### Case Study 2
In the second case study, a distillation column is added. The objective function is then changed to track the distillate concentration of B ($x_{D,B}$) by manipulating the reflux ratio ($R_R$) and the reactor jacket temperature ($T_j$). The distillation column is assumed to be at a constant temperature. This model is adapted from xxx. The simulation of this case study with the RL-PID, multiloop relay tuning, DFO-PID and NMPC control methods is shown below.
<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS2-1.png" width="100%">
</P>


The parametric mismatch analysis is shown below.

<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS2_ParaMis_UA-1.png" width="50%">
</P>

<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS2_ParaMis_k0-1.png" width="50%">
</P>

### Case Study 3
In the third case study, a recycle is added to the process used in Case Study 2. The objective function and manipulated variables remain the same as in Case Study 2. The simulation of this case study with the RL-PID, multiloop relay tuning, DFO-PID and NMPC control methods is shown below. The simulation of this case study with the RL-PID, multiloop relay tuning, DFO-PID and NMPC control methods is shown below.

<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS3-1.png" width="100%">
</P>

The parametric mismatch analysis is shown below.
<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS3_ParaMis_UA-1.png" width="50%">
</P>

<p align="center">
  <img src="https://github.com/MaximilianB2/MScProject/blob/main/fig/CS3_ParaMis_k0-1.png" width="50%">
</P>



