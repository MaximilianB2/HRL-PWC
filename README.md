# Hierarchical Reinforcement Learning for Plant-wide Control
Written by Max Bloor and supervised by Dr Antonio del Rio Chanona and Niki Kotecha

## Abstract
Proportional-Integral-Derivative (PID) controllers provide a simple and effective control solution used widely across the chemical industry. Their parameters require tuning with consideration of the entire system's dynamics to ensure adequate control. Traditional methods rely on system models or rules to tune these parameters. In this paper, a hierarchical reinforcement learning algorithm designed for plant-wide control is presented, consisting of a high-level artificial neural network and low-level PID controllers using an evolutionary algorithm for policy optimisation. Three case studies including a reactor, reactor-separator and reactor-separator-recycle systems are used to test the algorithm's performance at setpoint tracking, noise and disturbance rejection. The hierarchical reinforcement learning algorithm is compared to derivative-free optimisation, multiloop relay tuning, and a nonlinear model predictive controller (NMPC) for each case study. Across all case studies, the hierarchical reinforcement learning algorithm has a lower integral square error than both PID tuning methodologies. However, the NMPC showed that manipulation of other units in the system could result in enhanced setpoint tracking performance that PID-based methods could not replicate. The robustness of all controllers is investigated with a parametric mismatch analysis, which replicates the fouling of the reactor cooling jacket and the degradation of the reactor catalyst. This showed that due to the hierarchical reinforcement learning algorithm's lack of dependence on an accurate model, it can outperform the NMPC when there is a plant-model mismatch.

## Run Case Studies

Case study code can be found in ''CS1.ipynb'', ''CS2.ipynb'' and ''CS3.ipynb''

