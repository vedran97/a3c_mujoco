from sim_v import *
from A2C_torch import *
from tqdm import tqdm
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

state_size = 2700
action_size = 3
agents = []
alphas = [0.01,0.01,0.01,0.01,0.01,0.01]
best_errors = [1e1000,1e1000,1e1000,1e1000,1e1000,1e1000]
controller_range = 6
episodes = 100000
settling = [0,0,0,0,0,0]

INIT_GAINS = [[200,0,10],[800,0,100],[400,0,70],[200,0,30],[80,10,30],[50,0,10]]
pid_controllers = [PIDController(kp, ki, kd) for (kp,ki,kd) in INIT_GAINS]

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        value = self.critic(x)
        policy_logits = self.actor(x)
        
        return F.softmax(policy_logits, dim=-1), value

class PIDENV:
    def __init__(self,controller,idx):
        self.idx = idx
        self.controller = controller
        self.done = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def step(self,action,epoch):
        self.controller.reset_controller()
        for controller in pid_controllers:
            controller.reset_controller()
        delta_kp = self.controller.alpha * action[0]
        delta_ki = self.controller.alpha * action[1]
        delta_kd = self.controller.alpha * action[2]
        self.controller.update_gains(delta_kp,delta_ki,delta_kd)
        pid_controllers[self.idx] = self.controller
        current_pos = simOnce(render=False,plot=False,pid_controllers=pid_controllers)
        error_sum = np.sum(np.abs(np.array(self.controller.tracking_errors, dtype=np.float32)))
        reward = -error_sum
        if error_sum < 0.005 * (state_size -1):
            reward = 10
            self.done = True
        next_state = np.array(current_pos)[:,self.idx]
        if epoch%10 == 0 :
            print("error sum:{} , Joint:{}".format(error_sum,self.idx+1))
            print('Gains: kp:{}, ki:{}, kd:{}'.format(self.controller.Kp,self.controller.Ki,self.controller.Kd))
        return next_state, reward, self.done, None
    def reset(self):
        return np.zeros(state_size)
    
def a2c(num_steps=1000000, gamma=0.99, lr=0.0007, beta=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pid_controllers[0].set_alpha(alphas[0])
    env = PIDENV(pid_controllers[0],0)
    num_inputs = state_size
    num_actions = 3
    hidden_size = 64
    model = ActorCritic(num_inputs, num_actions, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    state = env.reset()
    episode_reward = 0
    log_probs = []
    values = []
    rewards = []
    masks = []
    entropy = 0
    
    for i in tqdm(range(num_steps)):
        state = torch.FloatTensor(state).to(device)
        policy, value = model(state.unsqueeze(0))
        dist = Categorical(policy.squeeze())
        action = dist.sample(torch.Size((1,3))).squeeze()
        next_state, reward, done, _ = env.step(action.cpu().detach().numpy(),i)
        log_prob = dist.log_prob(action)
        
        episode_reward += reward
        
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(torch.tensor([reward], dtype=torch.float))
        masks.append(torch.tensor([1-done], dtype=torch.float))
        
        if done:
            state = env.reset()
            print(f"Episode {i} reward: {episode_reward}")
            episode_reward = 0
            print('Gains: kp:{}, ki:{}, kd:{}'.format(env.controller.Kp,env.controller.Ki,env.controller.Kd))
            break
        else:
            state = next_state
            
        if i % 100 == 0:
            print(f"Step {i}")
        
        if len(rewards) == 64:
            _, next_value = model(torch.FloatTensor(next_state).unsqueeze(0).to(device))
            returns = compute_returns(next_value.cpu(), rewards, masks, gamma)
            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns).cpu()
            values = torch.cat(values).cpu()
            advantage = returns - values
            actor_loss = -(log_probs.cpu() * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            entropy_loss = beta * (policy * torch.log(policy + 1e-10)).sum(dim=1).mean()

            loss = actor_loss + 0.5 * critic_loss - entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            log_probs = []
            values = []
            rewards = []
            masks = []
            

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

a2c(10000)