import queue

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from impala import Config, Counter, test_policy
from impala import Policy, ValueFn 


class Learner:
    def __init__(
        self,
        id: int,
        config: Config,
        policy: Policy,
        value_fn: ValueFn,
        q: mp.Queue,
        update_counter: Counter,
        timeout: int = 200,
        device: torch.device = torch.device("cpu")
    ):
        self.id = id
        self.device = device
        self.config = config
        self.policy = policy
        self.value_fn = value_fn

        self.optimizer = Adam([*self.policy.parameters(), *self.value_fn.parameters()], lr=self.config.lr)

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.99)

        self.timeout = timeout
        self.update_counter = update_counter
        self.q = q
        self.completion = mp.Event()

        self.p = mp.Process(target=self._learn, name=f"learner_{id}")
        print(f"[main] learner_{self.id} Initialized")
    
    def start(self):
        self.completion.clear()
        self.p.start()
        print(f"[main] Started learner_{self.id} with pid {self.p.pid}")
        
    def terminate(self):
        self.p.terminate()
        print(f"[main] Terminated learner_{self.id}")

    def join(self):
        self.p.join()

    def _learn(self):
        try:
            update_count = 0

            if self.config.verbose:
                print(f"[learner_{self.id}] Beginning Update_{update_count + 1}")

            while update_count < self.config.max_updates:
                
                traj_count = 0
                value_loss = 0.
                policy_loss = 0.
                policy_entropy = 0.
                loss = torch.zeros(1, dtype=torch.float32, device=self.device, requires_grad=True)
                reward = 0.

                while traj_count < self.config.batch_size:
                    try:
                        traj = self.q.get(timeout=self.timeout)
                    except queue.Empty as e:
                        print(
                            f"[learner_{self.id}] No trajectory recieved for {self.timeout}"
                            f" seconds. Exiting!"
                        )
                        self.completion.set()
                        raise e
                
                    print(f"[learner_{self.id}] Processing traj_{traj.id}")
                    traj_len = len(traj.r)
                    obs = torch.stack(traj.obs)
                    actions = torch.stack(traj.a)
                    r = torch.stack(traj.r)
                    reward += torch.sum(r).item() / self.config.batch_size
                    disc = self.config.gamma * (~torch.stack(traj.d))

                    v = self.value_fn(obs).squeeze(1)
                    curr_logits = self.policy(obs[:-1])
                    
                    curr_log_probs = -F.nll_loss(torch.log_softmax(curr_logits, dim=-1), 
                                                torch.flatten(actions), reduction="none").view_as(actions)
                    
                    traj_log_probs = -F.nll_loss(torch.log_softmax(torch.stack(traj.logits), dim=-1), 
                                                torch.flatten(actions), reduction="none").view_as(actions)

                    with torch.no_grad():
                        imp_sampling = torch.exp(curr_log_probs - traj_log_probs).squeeze(1)
                        rho = torch.clamp(imp_sampling, max=self.config.rho_bar)
                        c = torch.clamp(imp_sampling, max=self.config.c_bar)
                        delta = rho * (r + self.config.gamma * v[1:] - v[:-1])
                        vt = torch.zeros(traj_len + 1, device=self.device, dtype=torch.float32)

                        for i in range(traj_len - 1, -1, -1):
                            vt[i] = delta[i] + disc[i] * c[i] * (vt[i + 1] - v[i + 1])
                        vt = torch.add(vt, v)

                        pg_adv = rho * (r + disc * vt[1:] - v[:-1])
                    
                    traj_value_loss = 0.5 * torch.sum((v - vt) ** 2)
                    cross_entropy = F.nll_loss(torch.log_softmax(curr_logits, dim=-1), 
                                            torch.flatten(actions), reduction="none").view_as(pg_adv)
                    traj_policy_loss = torch.sum(cross_entropy * pg_adv.detach())
                    policy = F.softmax(curr_logits, dim=-1)
                    log_policy = F.log_softmax(curr_logits, dim=-1)
                    traj_policy_entropy = -torch.sum(policy * log_policy)

                    traj_loss = (self.config.value_loss_coef * traj_value_loss
                                + self.config.policy_loss_coef * traj_policy_loss
                                - self.config.entropy_coef * traj_policy_entropy)
                    
                    loss = torch.add(loss, traj_loss / self.config.batch_size)
                    value_loss = torch.add(value_loss, traj_value_loss / self.config.batch_size)
                    policy_loss = torch.add(policy_loss, traj_policy_loss / self.config.batch_size)
                    policy_entropy = torch.add(policy_entropy, traj_policy_entropy / self.config.batch_size)
                    traj_count += 1                    

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.config.max_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.value_fn.parameters(), self.config.max_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                
                if self.config.verbose:
                    print(
                        f"[learner_{self.id}] Update {update_count + 1} | "
                        f"Batch Mean Reward: {reward:.2f} | Loss: {loss.item():.2f}"
                    )

                if self.config.eval_interval is not None:
                    if (update_count + 1) % self.config.eval_interval == 0:
                        eval_r, eval_std = test_policy(
                            self.policy,
                            self.config.env_name,
                            self.config.eval_episodes,
                            deterministic=True,
                            max_episode_len=self.config.max_timesteps,
                            device=self.device
                        )
                        if self.config.verbose:
                            print(
                                f"[learner_{self.id}] Update {update_count + 1} | "
                                f"Evaluation Reward: {eval_r:.2f}, Std Dev: {eval_std:.2f}"
                            )
                self.update_counter.increment()
                update_count = self.update_counter.value
            
            print(f"[learner_{self.id}] Finished learning")
            self.completion.set()
            return
        
        except KeyboardInterrupt:
            print(f"[learner_{self.id}] Interrupted")
            self.completion.set()
            return

        except Exception as e:
            print(f"[learner_{self.id}] Encoutered exception")
            raise e
        
    def save(self, path):
        """ Save model parameters """
        torch.save(
            {
                "actor_state_dict": self.policy.state_dict(),
                "critic_state_dict": self.value_fn.state_dict(),
            },
            path,
        )

    def load(self, path):
        """ Load model parameters """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["actor_state_dict"])
        self.value_fn.load_state_dict(checkpoint["critic_state_dict"])

    @property
    def policy_weights(self) -> torch.Tensor:
        return self.policy.state_dict()
    