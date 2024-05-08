from a3c import ActorCritic, SharedAdam, Worker

import torch
import gym
import torch.multiprocessing as mp 
from torch.multiprocessing import set_start_method
import os   
try:
    set_start_method('spawn')
except RuntimeError:
    pass


if __name__ == "__main__":

    env_name = 'Pendulum-v1'
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    global_net = ActorCritic(state_dim, action_dim, action_scale=2.)
    global_net.share_memory()
    optimizer = SharedAdam(global_net.parameters(), lr=1e-4, betas=(0.95, 0.999))

    global_episode, global_episode_return, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    workers = [Worker(env_name, global_net, optimizer, global_episode, global_episode_return, res_queue, i) for i in range(mp.cpu_count() // 4)]
    [w.start() for w in workers]
    res = []
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()