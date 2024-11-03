from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.conversions import aec_to_parallel
from pettingzoo.utils import ParallelEnv
import supersuit as ss
from stable_baselines3 import DQN
import numpy as np
import gymnasium as gym
from pettingzoo.utils.wrappers import BaseParallelWrapper


class RewardModifierWrapper(BaseParallelWrapper):
    def __init__(self, env, reward_factor: float = 1.0, penalty: float = -10.0, max_steps: int = 1000):
        super().__init__(env)
        self.env = env
        self.REWS = None
        self.DONES = None
        self.penalty = penalty      # Pénalité appliquée en cas de fin prématurée
        self.reward_factor = reward_factor  # Facteur multiplicatif pour les récompenses
        self.current_step = 0       # Compteur de pas actuel
        self.max_steps = max_steps       # Nombre maximal de pas par épisode
        self.nb_morts = 0
        self.nb_agents = 4

    def reset(self, seed=None, options=None):
        # Réinitialiser l'environnement et le compteur de pas
        self.current_step = 0
        self.nb_morts = 0
        print(f"Resetting environment {self.REWS=} {self.DONES=}")
        return self.env.reset(seed=seed, options=options)

    def step(self, actions):
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        obss, rews, terms, truncs, infos = self.env.step(active_actions)
        self.current_step += 1
        self.DONES = terms
        # print(f"{truncs=}, {terms=}")
        # Appliquer la pénalité si truncs est vrai (fin prématurée) 
        print(f"{truncs=}")
        for agent in self.env.agents:
            if truncs[agent]:
                rews[agent] += self.penalty
                self.nb_morts += 1
                print(f"Agent {agent} a été pénalisé pour fin prématurée")

            # Test si tous les agents sont morts
            if self.nb_morts == self.nb_agents:
                print(f"Tous les agents sont morts. Réinitialisation de l'environnement.")
                self.REWS = rews
                obss = self.reset()
  
        
        # Si le nombre maximal de pas est atteint, reset
        # obss = self.reset() if self.current_step >= self.max_steps else obss
        self.REWS = rews

        return obss, rews, terms, truncs, infos



import gymnasium as gym
import numpy as np
from pettingzoo.utils.wrappers import BaseParallelWrapper

from supersuit.utils.wrapper_chooser import WrapperChooser


class Black_death_par_new(BaseParallelWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.agents =           ["archer_0", "archer_1", "knight_0", "knight_1"]
        self.possible_agents =  ["archer_0", "archer_1", "knight_0", "knight_1"]

    def _check_valid_for_black_death(self):
        for agent in self.agents:
            space = self.observation_space(agent)
            assert isinstance(
                space, gym.spaces.Box
            ), f"observation sapces for black death must be Box spaces, is {space}"

    def reset(self, seed=None, options=None):
        obss, infos = self.env.reset(seed=seed, options=options)
        # self.agents = self.env.agents[:]
        self._check_valid_for_black_death()
        black_obs = {
            agent: np.zeros_like(self.observation_space(agent).low)
            for agent in self.agents
            if agent not in obss
        }
        total_obs = {**black_obs, **obss}
        # print(f"Resetting environment {total_obs=}, {type(total_obs)=}")
        return total_obs, infos

    def step(self, actions):
        active_actions = {agent: actions[agent] for agent in self.env.agents}
        obss, rews, terms, truncs, infos = self.env.step(active_actions)
        black_obs = {
            agent: np.zeros_like(self.observation_space(agent).low)
            for agent in self.agents
            if agent not in obss
        }
        black_rews = {agent: 0.0 for agent in self.agents if agent not in obss}
        black_infos = {agent: {} for agent in self.agents if agent not in obss}
        black_terms = {agent: False for agent in self.agents if agent not in obss}
        terminations = {**black_terms, **terms}
        te = np.fromiter(terms.values(), dtype=bool)
        tr = np.fromiter(truncs.values(), dtype=bool)
        env_is_done = (te & tr).all()
        total_obs = {**black_obs, **obss}
        total_rews = {**black_rews, **rews}
        total_infos = {**black_infos, **infos}

        if env_is_done:
            self.agents.clear()
        return total_obs, total_rews, terminations, terminations, total_infos

