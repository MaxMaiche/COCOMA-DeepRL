from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel
import supersuit as ss
from stable_baselines3 import DQN
import numpy as np
import matplotlib.pyplot as plt
import time 
from wrapper import  Black_death_par_new, RewardModifierWrapper
from deleteLog import deleteLog
from supersuit.vector import MarkovVectorEnv


if __name__ == "__main__":
    # Charger l'environnement PettingZoo AEC (par défaut)
    aec_env = knights_archers_zombies_v10.env()

    # Appliquer le wrapper 'RewardWrapper' pour gérer les agents inactifs
    # aec_env = RewardWrapper(aec_env)

    # Convertir l'environnement AEC en ParallelEnv
    parallel_env = aec_to_parallel(aec_env)

    # Appliquer le wrapper 'black_death_v3' pour gérer les agents inactifs
    parallel_env = Black_death_par_new(parallel_env)

    parallel_env = RewardModifierWrapper(parallel_env)

    # Utiliser SuperSuit pour convertir l'environnement en un environnement Gym compatible
    # gym_env = ss.pettingzoo_env_to_vec_env_v1(parallel_env)
    gym_env = MarkovVectorEnv(parallel_env, black_death=True)
    print(f"{gym_env.black_death=}")
    gym_env = ss.concat_vec_envs_v1(gym_env, 1, 1, base_class='stable_baselines3')
    # Initialiser les modèles DQN pour chaque agent
    num_agents = gym_env.num_envs  # Récupérer le nombre d'agents à partir de l'environnement

    policy_kwargs = dict(
        net_arch=[64]
    )

    models_archer = [DQN(  
                    'MlpPolicy', 
                    gym_env, 
                    verbose=0,
                    learning_rate=0.001,
                    buffer_size=2000,
                    batch_size=64,
                    learning_starts=200,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=100,
                    exploration_fraction=0.3,
                    exploration_initial_eps=0.05,
                    exploration_final_eps=0.03,
                    gamma=0.96,
                    policy_kwargs=policy_kwargs
                ) for _ in range(2)]

    models_knight = [DQN(  
                    'MlpPolicy', 
                    gym_env, 
                    verbose=0,
                    learning_rate=0.001,
                    buffer_size=2000,
                    batch_size=64,
                    learning_starts=200,
                    train_freq=1,
                    gradient_steps=1,
                    target_update_interval=100,
                    exploration_fraction=0.3,
                    exploration_initial_eps=0.05,
                    exploration_final_eps=0.05,
                    gamma=0.96,
                    policy_kwargs=policy_kwargs
                ) for _ in range(2)]

    models = models_archer + models_knight

    
    start_t = time.time()

    # Entraîner les agents
    timesteps = 1_000
    nb_episodes = 1
    for episode in range(nb_episodes):  # Nombre d'épisodes d'entraînement
        t = time.time() - start_t
        hours, remainder = divmod(t, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Episode {episode + 1} - Temps ecoule: {int(hours):02}:{int(minutes):02}:{int(seconds):02} s")
        obs = gym_env.reset()
        done = np.array([False] * num_agents)

        for step in range(timesteps):
            # if step % 1000 == 0:
            #     print(f"Step {step}/{timesteps}")
            # actions = np.array([models[i].predict(obs[i])[0] for i in range(num_agents)])  # Prédire les actions pour chaque agent
            # obs, rewards_batch, done, _ = gym_env.step(actions)  # Appliquer les actions dans l'environnement

            for i in range(num_agents):
                models[i].learn(total_timesteps=1, reset_num_timesteps=False)  # Mettre à jour chaque agent après chaque étape
                # Set black_death = True
                gym_env.venv.black_death = True

            # if done.all():
            #     # print(f"Tous les agents sont morts. Reinitialisation de l'environnement. {done=} {rewards_batch=}")
            #     obs = gym_env.reset()
            #     done = np.array([False] * num_agents)

        # for i in range(num_agents):
        #     models[i].learn(total_timesteps=100)

        # Sauvegarder les modèles
        # for i in range(num_agents):
        #     models[i].save(f"./checkpoints/dqn_agent_{i + 1}_knights_archers_zombies_{episode}")

        
        deleteLog()

    # Sauvegarder les modèles
    for i in range(num_agents):
        models[i].save(f"dqn_agent_{i + 1}_knights_archers_zombies")

    print("Entraînement terminé !")





    """
    ### TODO TRAIN AGENT INDENPENDENTLY FIRST THEN TRAIN THEM TOGETHER

    # Train them alone 

    # start_t = time.time()
    # timesteps = 2_000
    # for i, agent in enumerate(models):
    #     for episode in range(10):
    #         t = time.time() - start_t
    #         hours, remainder = divmod(t, 3600)
    #         minutes, seconds = divmod(remainder, 60)
    #         print(f"Episode {episode + 1} - Temps ecoule: {int(hours):02}:{int(minutes):02}:{int(seconds):02} s")
    #         obs = gym_env.reset()
    #         done = np.array([True] * num_agents)
    #         done[i] = False

    #         for step in range(timesteps):
    #             if step % 1000 == 0:
    #                 print(f"Step {step}/{timesteps}")
    #             actions = np.array([models[i].predict(obs[i])[0] for i in range(num_agents)])
    #             obs, rewards_batch, done, _ = gym_env.step(actions)  # Appliquer les actions dans l'environnement
    #             models[i].learn(total_timesteps=1)  # Mettre à jour chaque agent après chaque étape

    #             if done.all(): # Si tous les agents sont morts (fin de l'épisode)
    #                 done = np.array([True] * num_agents)
    #                 done[i] = False

    #         deleteLog()

    # # Save the models
    # for i in range(num_agents):
    #     models[i].save(f"ALONE_dqn_agent_{i + 1}_knights_archers_zombies")
    """