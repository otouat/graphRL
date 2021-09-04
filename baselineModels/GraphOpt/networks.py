import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, name='critic', fcl_dims=256, fc2_dims=256,
                 checkpoint_dir='tmp\\sac'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fcl_dims = fcl_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.net = nn.Sequential(
            nn.Linear(self.input_dims[0] + self.n_actions, self.fcl_dims),
            nn.ReLU(),
            nn.Linear(self.fcl_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, input_dims, fcl_dims=256, fc2_dims=256, name='value', checkpoint_dir='tmp\\sac'):
        super(ValueNetwork, self).__init__()
        self.input_dims = input_dims
        self.fcl_dims = fcl_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.net = nn.Sequential(
            nn.Linear(*self.input_dims, self.fcl_dims),
            nn.ReLU(),
            nn.Linear(self.fcl_dims, self.fc2_dims),
            nn.ReLU(),
            nn.Linear(self.fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        return self.net(state)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, max_action, fc1_dims=256, fc2_dims=256, n_actions=2, name='actor',
                 checkpoint_dir='tmp\\sac'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fcl_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        self.max_action = max_action
        self.reparameterisation_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fcl_dims)
        self.fc2 = nn.Linear(self.fcl_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = torch.clamp(sigma, min=self.reparameterisation_noise, max=1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1 - action.pow(2) + self.reparameterisation_noise)
        log_probs = log_probs.sum(1, keepdims=True)

        return action, log_probs

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))


class CostNetwork(nn.Module):
    def __init__(self, reward_lr, input_dims, hidden_dims=256, name='reward', checkpoint_dir='tmp\\sac'):
        super(CostNetwork, self).__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.net = nn.Sequential(
            nn.Linear(self.input_dims, self.hidden_dims),
            F.ReLU(),
            nn.Linear(self.hidden_dims, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=reward_lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, x):
        return self.net(x)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
