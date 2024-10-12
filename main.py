from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt

# Charger l'environnement PettingZoo AEC (par défaut)
aec_env = knights_archers_zombies_v10.env()

# Convertir l'environnement AEC en ParallelEnv
parallel_env = aec_to_parallel(aec_env)

# Appliquer le wrapper 'black_death_v3' pour gérer les agents inactifs
parallel_env = ss.black_death_v3(parallel_env)

# Utiliser SuperSuit pour convertir l'environnement en un environnement Gym compatible
gym_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
gym_env = ss.concat_vec_envs_v1(gym_env, 1, base_class='stable_baselines3')

# Initialiser le modèle DQN
model = DQN('MlpPolicy', gym_env, verbose=1)

# Entraîner l'agent
model.learn(total_timesteps=10000)

# Sauvegarder le modèle
model.save("dqn_knights_archers_zombies")

# Charger le modèle pour l'utiliser ou l'évaluer
model = DQN.load("dqn_knights_archers_zombies")

# Évaluer le modèle
env = knights_archers_zombies_v10.parallel_env()
env = ss.black_death_v3(env)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
model.set_env(env)

obs = env.reset()
rewards = []
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    rewards.append(reward)
    if done.any():
        obs = env.reset()

# Afficher les récompenses
plt.plot(np.cumsum(rewards))
plt.show()

# # Make a video of the trained model
# env = knights_archers_zombies_v10.parallel_env()
# env = ss.black_death_v3(env)
# env = ss.pettingzoo_env_to_vec_env_v1(env)
# env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
# model.set_env(env)
# ss.record_video(env, model, "dqn_knights_archers_zombies.mp4", video_length=1000, fps=10)

# # Afficher la vidéo
