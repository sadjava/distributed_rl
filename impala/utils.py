from dataclasses import dataclass
from typing import List, Union
import datetime

import torch
import torch.multiprocessing as mp
import gym
import numpy as np

from impala import Policy


@dataclass
class Config:
    max_updates: int = 50
    hidden_size: int = 128
    batch_size: int = 32
    gamma: float = 0.99
    rho_bar: float = 1.0
    c_bar: float = 1.0
    lr: float = 1e-3
    policy_loss_coef: float = 1.0
    value_loss_coef: float = 0.5
    entropy_coef: float = 5e-4
    max_timesteps: int = 1000
    queue_limit: int = 8
    max_norm: float = 10
    n_actors: int = 4
    env_name: str = 'CartPole-v1'
    log_path: str = './logs/'    
    save_interval: int = 50
    eval_interval: int = 2
    eval_episodes: int = 20
    verbose: bool = True
    render: bool = False

class Counter:
    def __init__(self, init_val: int = 0):
        self._val = mp.RawValue('i', init_val)
        self._lock = mp.Lock()
    
    def increment(self):
        with self._lock:
            self._val.value += 1
        
    @property
    def value(self):
        with self._lock:
            return self._val.value


class Trajectory:
    def __init__(
        self,
        id: int,
        observations: List[torch.Tensor] = [],
        actions: List[torch.Tensor] = [],
        rewards: List[torch.Tensor] = [],
        dones: List[torch.Tensor] = [],
        logits: List[torch.Tensor] = [],
    ):
        self.id = id
        self.obs = observations
        self.a = actions
        self.r = rewards
        self.d = dones
        self.logits = logits
    
    def add(
        self,
        obs: torch.Tensor,
        a: torch.Tensor,
        r: torch.Tensor,
        d: torch.Tensor,
        logits: torch.Tensor,
    ):
        self.obs.append(obs)
        self.a.append(a)
        self.r.append(r)
        self.d.append(d)
        self.logits.append(logits)

                
def make_env(env_name: str):
    env = gym.make(env_name)
    assert env.action_space.__class__.__name__ == 'Discrete', "Only discrete action space is supported"
    return env

def test_policy(
    policy: Policy,
    env: Union[gym.Env, str],
    episodes: int,
    deterministic: bool,
    max_episode_len: int,
    verbose: bool = False,
    device: torch.device = torch.device('cpu'),
):  
    start_time = datetime.datetime.now()
    if verbose:
        print(f'Starting test at {start_time:%d-%m-%Y %H:%M:%S}')
    if type(env) == str:
        env = make_env(env)
    
    policy.eval()
    rewards = []
    for e in range(episodes):
        obs = env.reset()
        obs = torch.tensor(obs, device=device, dtype=torch.float32)
        d = False
        ep_rewards = []
        for t in range(max_episode_len):
            action, _ = policy.get_action(obs, deterministic=deterministic)
            obs, reward, d, _ = env.step(action.item())
            obs = torch.tensor(obs, device=device, dtype=torch.float32)
            ep_rewards.append(reward)
            if d:
                break
        rewards.append(sum(ep_rewards))
    avg_reward = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print(f"Testing completed in {(datetime.datetime.now() - start_time).seconds}\nAverage reward = {avg_reward:.2f} +/- {std:.2f}")
    return avg_reward, std                 
        
