import numpy as np
import matplotlib.pyplot as plt

numbers = ['zero','one','two','three','four','five','six','seven','eight','nine']

t = np.linspace(0,10,1000)
y = np.empty((1000,7))
for i in range(7):
  y[:,i] = np.sin(t)+np.random.uniform(-1,2,1000)
fig, ax = plt.subplots()
ax.plot(t,y, color = 'tab:red')
ax.fill_between(t,np.max(y,axis = 1),np.min(y,axis = 1), color = 'tab:red',alpha = 0.2)
plt.show()
