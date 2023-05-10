import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class GenericNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(GenericNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma=0.99, n_actions=36,
                 layer1_size=256, layer2_size=256, n_outputs=18):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size,
                                           layer2_size, n_actions=n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size,
                                            layer2_size, n_actions=1)

    def choose_action(self, observation):
        output =  self.actor.forward(observation).reshape(2,-1)
        #print('ACTOR OUTPUT',output)
        mu, sigma  = output[0],output[1]
        sigma = T.exp(sigma)
        action_probs = T.distributions.MultivariateNormal(mu, T.diag_embed(sigma))
        #print('ACTION PROBS', action_probs)
        probs = action_probs.sample()
        #print('PROBS',probs)
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.sigmoid(probs)
        #print('ACTION',action)

        return action.cpu().detach().numpy()

    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        print('Critic Value:',critic_value.item())
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = ((reward + self.gamma*critic_value_*(1-int(done))) - \
                                                                critic_value)
        
        #print('LOSS:',delta.item()**2,'= ',reward.item(),'(REWARD) +',self.gamma*critic_value_.item(),'- ',critic_value.item() )
        

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

