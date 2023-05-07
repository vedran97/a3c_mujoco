import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size,actor_hidden_size=24):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.actor_hidden_size = actor_hidden_size
        self.d1 = nn.Linear(state_size, self.actor_hidden_size)
        self.d2 = nn.Linear(self.actor_hidden_size, self.actor_hidden_size)

        self.out_mu = nn.Linear(self.actor_hidden_size, action_size)
        self.out_std = nn.Linear(self.actor_hidden_size, action_size)

    def forward(self, input_state):
        d1 = nn.functional.relu(self.d1(input_state))
        d2 = nn.functional.relu(self.d2(d1))
        out_mu = torch.tanh(self.out_mu(d2)) # tanh / linear
        out_std = nn.functional.softplus(self.out_std(d2))
        return out_mu, out_std

class Critic(nn.Module):
    def __init__(self, state_size,critic_hidden_size=24):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.critic_hidden_size = critic_hidden_size
        self.d1 = nn.Linear(state_size, critic_hidden_size)
        self.d2 = nn.Linear(critic_hidden_size, critic_hidden_size)
        self.output = nn.Linear(critic_hidden_size, 1)

    def forward(self, input_state):
        d1 = nn.functional.relu(self.d1(input_state))
        d2 = nn.functional.relu(self.d2(d1))
        output = self.output(d2)
        return output

class A2CPIDTuner:
    def __init__(self, state_size, action_size, load_model=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model = load_model
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1
        self.grad_bound = 0
        
        self.std_bound = [0.01, 1]


        self.discount_factor = 0.95
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self.actor = Actor(state_size, action_size,actor_hidden_size=500).to(self.device)
        self.critic = Critic(state_size,critic_hidden_size=500).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def log_pdf(self, mu, std, action):
        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        var = std**2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * torch.log(var * 2 * np.pi)
        return torch.sum(log_policy_pdf, dim=1, keepdim=True)
    
    def get_action(self, state):
        mu, std = self.actor(state)

        mu = mu[0]
        std = std[0]

        std = torch.clamp(std, self.std_bound[0], self.std_bound[1])
        action = np.random.normal(mu.cpu().detach().numpy(), std.cpu().detach().numpy(), size=self.action_size)
        return action
    
    def save_model(self):
        torch.save(self.actor.state_dict(), "./actor_trained.pth")
        torch.save(self.critic.state_dict(), "./critic_trained.pth")

    def train_actor(self, action, state, advantage):
        mu_a, std_a = self.actor(state)
        log_policy_pdf = self.log_pdf(mu_a, std_a, action)
        loss = torch.sum(-log_policy_pdf * advantage)

        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_bound)
        self.actor_optimizer.step()

    def train_critic(self, state, target):
        output = self.critic(state)
        target = target.reshape(output.shape)
        loss2 = F.mse_loss(output, target)

        self.critic_optimizer.zero_grad()
        grad = loss2.backward()
        print(grad)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_bound)
        self.critic_optimizer.step()

    def train_model(self, state, action, reward, next_state, done):
        value = self.critic(state)[0]
        next_value = self.critic(next_state)[0]

        # Bellman expectation equation update
        advantage = reward - value + (1 - done)*(self.discount_factor * next_value)
        target = reward + (1 - done)*(self.discount_factor * next_value)
        
        action = torch.tensor(action, dtype=torch.float).to(self.device)
        # self.train_critic(state, target)
        # self.train_actor(action, state, advantage)

        ## train actor:
        mu_a, std_a = self.actor(state)
        log_policy_pdf = self.log_pdf(mu_a, std_a, action)
        loss = torch.sum(-log_policy_pdf * advantage)
        loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_bound)

        ##train critic:
        output = self.critic(state)
        target = target.reshape(output.shape)
        loss2 = F.mse_loss(output, target)
        loss2.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_bound)

        self.actor_optimizer.step()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()