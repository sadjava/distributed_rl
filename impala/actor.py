import queue

import torch
import torch.multiprocessing as mp

from impala import Config, Counter, Trajectory, make_env
from impala import Policy
from impala import Learner

class Actor:
    def __init__(
        self,
        id: int,
        config: Config,
        policy: Policy,
        learner: Learner,
        q: mp.Queue,
        update_counter: Counter,
        timeout: int = 10,
        device: torch.device = torch.device("cpu")
    ):
        self.id = id
        self.device = device
        self.config = config
        self.policy = policy
        for p in self.policy.parameters():
            p.requires_grad = False
        self.learner = learner
        self.timeout = timeout
        self.q = q
        self.update_counter = update_counter
        self.completion = mp.Event()
        self.p = mp.Process(target=self._act, name=f"actor_{id}")
        print(f"[main] actor_{id} Initialized")
    
    def start(self):
        self.p.start()
        print(f"[main] actor_{self.id} Started with pid {self.p.pid}")

    def terminate(self):
        self.p.terminate()
        print(f"[main] actor_{self.id} Terminated")
                
    def join(self):
        self.p.join()

    def _act(self):
        try:

            env = make_env(self.config.env_name)
            traj_no = 0

            while not self.learner.completion.is_set():
                traj_no += 1
                self.policy.load_state_dict(self.learner.policy_weights)
                traj_id = (self.id, traj_no)
                traj = Trajectory(traj_id, [], [], [], [], [])
                obs = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                traj.obs.append(obs)
                c = 0

                print(f"[actor_{self.id}] Starting traj_{traj.id}")

                while c < self.config.max_timesteps:
                    if self.config.render:
                        env.render()
                    c += 1
                    a, logits = self.policy.get_action(obs)

                    obs, r, d, _ = env.step(a.item())
                    obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
                    r = torch.tensor(r, dtype=torch.float32, device=self.device)
                    d = torch.tensor(d, device=self.device)
                    traj.add(obs, a, r, d, logits)

                    if d:
                        break 
                
                if self.config.verbose:
                    print(f"[actor_{self.id}] traj_{traj.id} completed Reward = {sum(traj.r)}")
            
                while True:
                    try:
                        self.q.put(traj, timeout=self.timeout)
                        break
                    except queue.Full:
                        if self.learner.completion.is_set():
                            break
                        else:
                            continue
            env.close()
            print(f"[actor_{self.id}] Finished acting")
            self.completion.set()
            return

        except KeyboardInterrupt:
            print(f"[actor_{self.id}] interrupted")
            env.close()
            self.completion.set()
            return

        except Exception as e:
            env.close()
            print(f"[actor_{self.id}] encoutered exception")
            raise e
