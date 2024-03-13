from stable_baselines3 import SAC
from CS1_Model import reactor_class
from stable_baselines3.common.callbacks import BaseCallback
# import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import CheckpointCallback

class RewardCallback(BaseCallback):
    def __init__(self, check_freq: int):
        super(RewardCallback, self).__init__()
        self.check_freq = check_freq
        self.rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          # Assuming that self.training_env is your training environment
          obs = self.training_env.reset()
          done = False
          rewards = []
          while not done:
              action, _ = self.model.predict(obs)
              obs, reward, done, info = self.training_env.step(action)
              rewards.append(reward)
          self.rewards.append(sum(rewards))
        return True

checkpoint_callback = CheckpointCallback(save_freq=100, save_path="./logs/SAC_vel",
                                         name_prefix="SAC_model_vel_1602")
env = reactor_class(test=False,ns = 240,PID_vel=True)
reward_callback = RewardCallback(check_freq=500)
model = SAC("MlpPolicy", env, verbose=1,learning_rate=1e-3,device = 'cuda')
# print(model.policy)
model.learn(int(3e4))
model.save('SAC_Vel_0403')
SAC_Training_Rewards = reward_callback.rewards

# def learning_curve_plot(r):
#   fig, axs = plt.subplots(1, 1, figsize=(5, 5))
#   plt.rcParams['text.usetex'] = 'True'
#   axs.plot(r,color = 'tab:blue')
#   axs.set_xlabel('Number of Time Steps')
#   axs.set_ylabel('Reward')
#   plt.savefig('SAC_Learning_Curve.pdf')
#   plt.show
# learning_curve_plot(SAC_Training_Rewards)