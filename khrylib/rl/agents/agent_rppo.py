import math
from khrylib.utils.torch import *
from khrylib.rl.agents import AgentPG
import torch.optim as optim


# Revised PPO
class AgentrPPO(AgentPG):

    def __init__(self, clip_epsilon=0.2, mini_batch_size=64, policy_grad_clip=None,
                 entropy_coef=0.01, gae_lambda=0.95, value_clip_epsilon=0.2,
                 lr_scheduler_step_size=1000, lr_scheduler_gamma=0.9, **kwargs):
        super().__init__(**kwargs)
        self.clip_epsilon = clip_epsilon
        self.mini_batch_size = mini_batch_size
        self.policy_grad_clip = policy_grad_clip
        self.entropy_coef = entropy_coef
        self.gae_lambda = gae_lambda
        self.value_clip_epsilon = value_clip_epsilon
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_scheduler_step_size,
                                                      gamma=lr_scheduler_gamma)

    def update_policy(self, states, actions, returns, advantages, exps, iteration):
        """update policy"""
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with to_test(*self.update_modules):
            with torch.no_grad():
                fixed_log_probs = self.policy_net.get_log_prob(self.trans_policy(states), actions)

        for _ in range(self.opt_num_epochs):
            perm = np.arange(states.shape[0])
            np.random.shuffle(perm)
            perm = LongTensor(perm).to(self.device)

            states, actions, returns, advantages, fixed_log_probs, exps = \
                states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), \
                    fixed_log_probs[perm].clone(), exps[perm].clone()

            optim_batch_num = int(math.floor(states.shape[0] / self.mini_batch_size))
            for i in range(optim_batch_num):
                ind = slice(i * self.mini_batch_size, min((i + 1) * self.mini_batch_size, states.shape[0]))
                states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b, exps_b = \
                    states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind], exps[ind]
                ind = exps_b.nonzero(as_tuple=False).squeeze(1)

                value_loss = self.value_loss(states_b, returns_b)
                surr_loss = self.ppo_loss(states_b, actions_b, advantages_b, fixed_log_probs_b, ind)
                entropy_loss = self.entropy_loss(states_b, ind)

                loss = surr_loss + self.value_pred_coef * value_loss - self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.clip_policy_grad()
                self.optimizer.step()

            self.lr_scheduler.step()

    def clip_policy_grad(self):
        if self.policy_grad_clip is not None:
            for params, max_norm in self.policy_grad_clip:
                torch.nn.utils.clip_grad_norm_(params, max_norm)

    def ppo_loss(self, states, actions, advantages, fixed_log_probs, ind):
        log_probs = self.policy_net.get_log_prob(self.trans_policy(states)[ind], actions[ind])
        ratio = torch.exp(log_probs - fixed_log_probs[ind])
        advantages = advantages[ind]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        surr_loss = -torch.min(surr1, surr2).mean()
        return surr_loss

    def value_loss(self, states, returns):
        values = self.value_net(self.trans_value(states))
        value_clipped = values + torch.clamp(values - returns, -self.value_clip_epsilon, self.value_clip_epsilon)
        value_loss = torch.max((values - returns).pow(2), (value_clipped - returns).pow(2)).mean()
        return value_loss

    def entropy_loss(self, states, ind):
        entropy = self.policy_net.get_entropy(self.trans_policy(states)[ind])
        entropy_loss = entropy.mean()
        return entropy_loss



