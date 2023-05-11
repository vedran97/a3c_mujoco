
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from copy import deepcopy
import numpy as np

class Buffer:
    def __init__(self, observationDim: int, actionDim: int, size: int = 1_000_000):
        # use a fixed-size buffer to prevent constant list instantiations
        self.states = np.zeros((size, observationDim))
        self.actions = np.zeros((size, actionDim))
        self.rewards = np.zeros(size)
        self.nextStates = np.zeros((size, observationDim))
        self.doneFlags = np.zeros(size)
        # use a pointer to keep track of where in the buffer we are
        self.pointer = 0
        # use current size to ensure we don't train on any non-existent data points
        self.currentSize = 0
        self.size = size

    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        nextState: np.ndarray,
        doneFlag: bool,
    ):
        # store all the data for this transition
        ptr = self.pointer
        self.states[ptr] = state
        self.actions[ptr] = action
        self.rewards[ptr] = reward
        self.nextStates[ptr] = nextState
        self.doneFlags[ptr] = doneFlag
        # update the pointer and current size
        self.pointer = (self.pointer + 1) % self.size
        self.currentSize = min(self.currentSize + 1, self.size)

    def getMiniBatch(self, size: int) -> dict:
        # ensure size is not bigger than the current size of the buffer
        size = min(size, self.currentSize)
        # generate random indices
        indices = np.random.choice(self.currentSize, size, replace=False)
        # return the mini-batch of transitions
        return {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "nextStates": self.nextStates[indices],
            "doneFlags": self.doneFlags[indices],
        }


class Actor(nn.Module):
    def __init__(self, state_dim=3,action_dim=1,learningRate=0.01):
        super(Actor, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor((state_dim, action_dim)))
        # print("STATE DIM ACTOR:",state_dim)
        # print("WEIGHT DIM ACTOR:",self.weight)
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def forward(self, state):
        return torch.matmul(state, torch.exp(self.weight)) #torch.exp() is used to make sure the weights are positive
    def init_weights(self,gains):
        kp,ki,kd = gains
        self.weight.data = torch.FloatTensor([np.log(kp),np.log(ki),np.log(kd)]).to(self.device)
    def grad_descent(self,loss:torch.Tensor,retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()

class Critic(nn.Module):
    def __init__(self, input_dim,learningRate=0.01):
        super(Critic, self).__init__()
        self.layer_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        # ReLU activation function
        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=learningRate)
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def grad_descent(self,loss:torch.Tensor,retain_graph=False):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=retain_graph)
        self.optimizer.step()
    
class Agent:
    def __init__(self,env,learningRate,gamma,tau):
        self.buffer = Buffer(env.observation_dim,env.action_dim)
        self.observation_dim = env.observation_dim
        self.action_dim = env.action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ## Actor critic networks
        self.actor = Actor(self.observation_dim,self.action_dim,learningRate).to(device=self.device)
        self.critic1 = Critic(self.observation_dim+self.action_dim,learningRate).to(device=self.device)
        self.critic2 = Critic(self.observation_dim+self.action_dim,learningRate).to(device=self.device)
        ## Target networks
        self.actor_target = Actor(3,1,learningRate).to(device=self.device)
        self.critic1_target = Critic(self.observation_dim+self.action_dim,learningRate).to(device=self.device)
        self.critic2_target = Critic(self.observation_dim+self.action_dim,learningRate).to(device=self.device)

    def getDetAction(self,observation:np.ndarray):
        actions = self.actor(torch.Tensor(observation).to(self.device))
        return actions.detach().cpu().numpy()
    
    def getNoiseAction(self,observation:np.ndarray,variance:float=1.5):
        deterministicAction = self.getDetAction(observation)
        std_dev = np.sqrt(variance)
        noise = np.random.normal(0,std_dev,deterministicAction.shape)
        return np.clip(deterministicAction + noise, -5000,5000)
        # return deterministicAction
    
    def computeQLoss(
        self, network, states: torch.Tensor, actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # compute the MSE of the Q function with respect to the targets
        QValues = torch.squeeze(network.forward(torch.hstack([states, actions]).float()))
        return torch.square(QValues - targets).mean()

    def computePolicyLoss(self, states: torch.Tensor):
        actions = self.actor.forward(states.float())
        actions = actions.unsqueeze(-1)
        # print(states)
        # print(actions)
        QValues = torch.squeeze(self.critic1.forward(torch.hstack([states, actions]).float()))
        return -QValues.mean()

    def updateTargetNetwork(self, targetNetwork, network):
        with torch.no_grad():
            for targetParameter, parameter in zip(targetNetwork.parameters(), network.parameters()):
                targetParameter.mul_(1 - self.tau)
                targetParameter.add_(self.tau * parameter)

    def computeTargets(
        self,
        rewards: torch.Tensor,
        nextStates: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        # print("nextstates:",nextStates.float())
        nextStates = nextStates.squeeze()
        targetActions = self.actor_target.forward(nextStates.float().squeeze())
        # compute targets
        # print("nextstates:",nextStates)
        # print("targetActions:",targetActions.unsqueeze(-1))
        targetActions = targetActions.unsqueeze(-1)
        if targetActions.shape == torch.Size([1]):
            use_hstack = True
        else:
            use_hstack = False
        # print("hstack:",torch.hstack([nextStates, targetActions]))
        if use_hstack:
            targetQ1Values = torch.squeeze(
                self.critic1_target.forward(torch.hstack([nextStates, targetActions]).float())
            )

            targetQ2Values = torch.squeeze(
                self.critic2_target.forward(torch.hstack([nextStates, targetActions]).float())
            )
        else:
            targetQ1Values = torch.squeeze(
                self.critic1_target.forward(torch.cat((nextStates, targetActions), dim=1).float())
            )
            targetQ2Values = torch.squeeze(
                self.critic2_target.forward(torch.cat((nextStates, targetActions), dim=1).float())
            )
        targetQValues = torch.minimum(targetQ1Values, targetQ2Values)
        return rewards + self.gamma * (1 - dones) * targetQValues
    
    def update(
        self,
        miniBatchSize: int,
        trainingSigma: float,
        trainingClip: float,
        updatePolicy: bool,
    ):
        # randomly sample a mini-batch from the replay buffer
        miniBatch = self.buffer.getMiniBatch(miniBatchSize)
        # create tensors to start generating computational graph
        states = torch.tensor(miniBatch["states"], requires_grad=True, device=self.device)
        actions = torch.tensor(miniBatch["actions"], requires_grad=True, device=self.device)
        rewards = torch.tensor(miniBatch["rewards"], requires_grad=True, device=self.device)
        nextStates = torch.tensor(
            miniBatch["nextStates"], requires_grad=True, device=self.device
        )
        dones = torch.tensor(miniBatch["doneFlags"], requires_grad=True, device=self.device)
        # compute the targets
        targets = self.computeTargets(
            rewards, nextStates, dones
        )
        # do a single step on each critic network
        Q1Loss = self.computeQLoss(self.critic1, states, actions, targets)
        self.critic1.grad_descent(Q1Loss, True)
        Q2Loss = self.computeQLoss(self.critic2, states, actions, targets)
        self.critic2.grad_descent(Q2Loss)
        if updatePolicy:
            # do a single step on the actor network
            policyLoss = self.computePolicyLoss(states)
            self.actor.grad_descent(policyLoss)
            # update target networks
            self.updateTargetNetwork(self.actor_target, self.actor)
            self.updateTargetNetwork(self.critic1_target, self.critic1)
            self.updateTargetNetwork(self.critic2_target, self.critic2)