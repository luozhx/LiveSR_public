import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from abr.Network import ActorNetwork, CriticNetwork


class A3C(object):
    def __init__(self, is_central, s_dim, action_dim, actor_lr=1e-4, critic_lr=1e-3):
        self.s_dim = s_dim
        self.a_dim = action_dim
        self.discount = 0.99
        self.entropy_weight = 0.5
        self.entropy_eps = 1e-6

        self.max_grad_norm = 0.5

        self.is_central = is_central
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(self.s_dim, self.a_dim).to(self.device)
        if self.is_central:
            # unify default parameters for tensorflow and pytorch
            self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr=actor_lr, alpha=0.9, eps=1e-10)
            self.actor_optimizer.zero_grad()

            self.critic = CriticNetwork(self.s_dim, self.a_dim).to(self.device)
            self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=critic_lr, alpha=0.9, eps=1e-10)
            self.critic_optimizer.zero_grad()
        else:
            self.actor.eval()

        self.loss_function = nn.MSELoss()

    def get_gradient(self, s_batch, a_batch, r_batch, terminal):
        s_batch = torch.cat(s_batch).to(self.device)
        a_batch = torch.LongTensor(a_batch).to(self.device)
        r_batch = torch.tensor(r_batch).to(self.device)
        R_batch = torch.zeros(r_batch.shape).to(self.device)

        R_batch[-1] = r_batch[-1]
        for t in reversed(range(r_batch.shape[0] - 1)):
            R_batch[t] = r_batch[t] + self.discount * R_batch[t + 1]

        with torch.no_grad():
            v_batch = self.critic.forward(s_batch).squeeze().to(self.device)
        advantages = R_batch - v_batch

        probability = self.actor.forward(s_batch)
        cate_dist = Categorical(probability)
        action_log_probs = cate_dist.log_prob(a_batch)
        dist_entropy = cate_dist.entropy()

        action_loss = torch.sum(action_log_probs * (-advantages))
        entropy_loss = -self.entropy_weight * torch.sum(dist_entropy)

        actor_loss = action_loss + entropy_loss
        actor_loss.backward()

        critic_loss = self.loss_function(R_batch, self.critic.forward(s_batch).squeeze())
        critic_loss.backward()
        # use the feature of accumulating gradient in pytorch

    def select_action(self, states):
        if not self.is_central:
            with torch.no_grad():
                probability = self.actor.forward(states)
                m = Categorical(probability)
                action = m.sample().item()
                return action

    def hard_update_actor_parameters(self, actor_net_params):
        for target_param, source_param in zip(self.actor.parameters(), actor_net_params):
            target_param.data.copy_(source_param.data)

    def update_network(self):
        # use the feature of accumulating gradient in pytorch
        if self.is_central:
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad()

    def get_actor_params(self):
        return list(self.actor.parameters())

    def get_critic_params(self):
        return list(self.critic.parameters())
