from copy import deepcopy

import torch.multiprocessing as mp
import torch
import gym

from a3c.utils import push_and_pull, record, v_wrap
from a3c.model import ActorCritic


class Worker(mp.Process):
    def __init__(self, env_name, global_net, optimizer, global_episode, global_episode_return,
                 res_queue, name, max_episode=3000, episode_len=200, update_global_iter=5, gamma=0.9):
        super(Worker, self).__init__()
        self.name = "w%i" % name
        self.global_episode = global_episode
        self.global_episode_return = global_episode_return
        self.res_queue = res_queue

        self.global_net = global_net
        self.optimizer = optimizer
        
        self.max_episode = max_episode
        self.episode_len = episode_len
        self.update_global_iter = update_global_iter
        self.gamma = gamma

        self.env = gym.make(env_name).unwrapped
        self.local_net = ActorCritic(global_net.state_dim, global_net.action_dim, global_net.hidden_sizes, action_scale=global_net.action_scale)

    def run(self):
        total_step = 1
        while self.global_episode.value < self.max_episode:
            state = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            episode_return = 0.
            for t in range(self.episode_len):
                if self.name == 'w0': 
                    self.env.render()
                action = self.local_net.choose_action(v_wrap(state[None, :]))
                next_state, reward, done, _ = self.env.step(action.clip(-2, 2))
                if t == self.episode_len - 1:
                    done = True
                episode_return += reward
                buffer_a.append(action)
                buffer_s.append(state)
                buffer_r.append((reward + 8.1) / 8.1)

                if total_step % self.update_global_iter == 0 or done:
                    push_and_pull(self.optimizer, self.local_net, self.global_net, done, next_state, 
                                  buffer_s, buffer_a, buffer_r, self.gamma)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    if done:
                        record(self.global_episode, self.global_episode_return, episode_return, self.res_queue, self.name)
                        break
                state = next_state
                total_step += 1
        
        self.res_queue.put(None)
                    