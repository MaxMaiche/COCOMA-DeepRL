from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Charger l'environnement PettingZoo AEC (par défaut)
aec_env = knights_archers_zombies_v10.env(render_mode="rgb_array")

# Convertir l'environnement AEC en ParallelEnv
parallel_env = aec_to_parallel(aec_env)

# Appliquer le wrapper 'black_death_v3' pour gérer les agents inactifs
parallel_env = ss.black_death_v3(parallel_env)

# Utiliser SuperSuit pour convertir l'environnement en un environnement Gym compatible
gym_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
gym_env = ss.concat_vec_envs_v1(gym_env, 1, base_class='stable_baselines3')

# Initialiser les modèles DQN pour chaque agent
num_agents = gym_env.num_envs  # Récupérer le nombre d'agents à partir de l'environnement

models = []
# load the models
for i in range(num_agents):
    models.append(DQN.load(f"dqn_agent_{i + 1}_knights_archers_zombies"))

# Évaluer les modèles
obs = gym_env.reset()
rewards = [[] for _ in range(num_agents)]  # Liste pour stocker les récompenses de chaque agent
done = np.array([False] * num_agents)
frames = []
for _ in range(1000):
    actions = np.array([models[i].predict(obs[i])[0] for i in range(num_agents)] if not done[i] else 0)  # Prédire les actions pour chaque agent
    obs, rewards_batch, done, _ = gym_env.step(actions)  # Appliquer les actions dans l'environnement

    frame = gym_env.render()  # Capturer la frame de l'environnement
    frames.append(frame)

    for i in range(num_agents):
        rewards[i].append(rewards_batch[i])  # Stocker les récompenses pour chaque agent

    if done.all(): # Si tous les agents sont morts (fin de l'épisode)
        obs = gym_env.reset()
        done = np.array([False] * num_agents)

# Afficher les récompenses cumulées pour chaque agent
for i in range(num_agents):
    plt.plot(np.cumsum(rewards[i]), label=f'Agent {i + 1}')
plt.xlabel('Étapes')
plt.ylabel('Récompense cumulée')
plt.legend()
plt.title('Récompenses cumulées par agent')
plt.show()


# Créer une vidéo à partir des frames capturées
imageio.mimsave("dqn_knights_archers_zombies.mp4", frames, fps=30)