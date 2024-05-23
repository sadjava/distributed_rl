import datetime

import torch
import torch.multiprocessing as mp

from impala import Config, Counter, make_env
from impala import Policy, ValueFn
from impala import Learner
from impala import Actor


config = Config(
    max_updates=50,
    hidden_size=128,
    batch_size=32,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
    lr=1e-3,
    policy_loss_coef=1.0,
    value_loss_coef=0.5,
    entropy_coef=5e-4,
    max_timesteps=1000,
    queue_limit=8,
    max_norm=10,
    n_actors=8,
    env_name='CartPole-v1',
    log_path='./logs/',
    save_interval=50,
    eval_interval=2,
    eval_episodes=20,
    verbose=True,
    render=False,
)

def train(config: Config):
    device = torch.device('cpu')
    start_time = datetime.datetime.now()

    print(f"[main] Start time: {start_time:%d-%m-%Y %H:%M:%S}")
    print(f"[main] {config}\n")

    mp.set_start_method('fork', force=True)
    q = mp.Queue(config.queue_limit)
    update_counter = Counter(init_val=0)
    env = make_env(config.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    policy = Policy(state_dim, action_dim, config.hidden_size, device=device)
    policy.share_memory()
    value_fn = ValueFn(state_dim, config.hidden_size)
    learner = Learner(
        1,
        config,
        policy,
        value_fn,
        q,
        update_counter,
        device=device
    )
    
    actors = []
    for i in range(config.n_actors):
        policy = Policy(state_dim, action_dim, config.hidden_size, device=device)
        actors.append(Actor(i + 1, config, policy, learner, q, update_counter, device=device))

    print(f"[main] Initialized")
    for a in actors:
        a.start()
    learner.start()

    learner.completion.wait()
    for a in actors:
        a.completion.wait()
    
    learner.terminate()
    for a in actors:
        a.terminate()
    
    learner.join()
    for a in actors:
        a.join()

    print(f"[main] Completed in {(datetime.datetime.now() - start_time).seconds} seconds")

if __name__ == '__main__':
    train(config)    