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

# Initialiser les modèles DQN pour chaque agent
num_agents = gym_env.num_envs  # Récupérer le nombre d'agents à partir de l'environnement
models = [DQN(  
                'MlpPolicy', 
                gym_env, 
                verbose=0,
                learning_rate=0.001,
                buffer_size=1000,
                batch_size=32,
                learning_starts=200,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=100,
                exploration_fraction=0.1,
                exploration_final_eps=0.02,
                gamma=0.99,

            ) for _ in range(num_agents)]

# Entraîner les agents
timesteps = 10_000
for episode in range(10):  # Nombre d'épisodes d'entraînement
    print(f"Episode {episode + 1}")
    obs = gym_env.reset()
    done = np.array([False] * num_agents)

    for step in range(timesteps):
        if step % 1000 == 0:
            print(f"Step {step}/{timesteps}")
        actions = np.array([models[i].predict(obs[i])[0] for i in range(num_agents)])  # Prédire les actions pour chaque agent
        obs, rewards_batch, done, _ = gym_env.step(actions)  # Appliquer les actions dans l'environnement
        
        for i in range(num_agents):
            models[i].learn(total_timesteps=1)  # Mettre à jour chaque agent après chaque étape
        
        if done.all():
            obs = gym_env.reset()
            done = np.array([False] * num_agents)

    for i in range(num_agents):
        models[i].learn(total_timesteps=1000)

# Sauvegarder les modèles
for i in range(num_agents):
    models[i].save(f"dqn_agent_{i + 1}_knights_archers_zombies")

print("Entraînement terminé !")