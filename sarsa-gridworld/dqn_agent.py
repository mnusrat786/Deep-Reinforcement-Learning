import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim=2, output_dim=4):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, actions, alpha=0.001, gamma=0.99, epsilon=1.0):
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=10000)
        self.batch_size = 32

    def act(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            return self.actions[torch.argmax(q_values).item()]

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_batch = torch.FloatTensor([s for s, _, _, _, _ in batch])
        next_batch = torch.FloatTensor([ns for _, _, _, ns, _ in batch])
        action_batch = [self.actions.index(a) for _, a, _, _, _ in batch]
        reward_batch = torch.FloatTensor([r for _, _, r, _, _ in batch])
        done_batch = torch.FloatTensor([float(d) for _, _, _, _, d in batch])

        q_vals = self.model(state_batch)
        next_q_vals = self.model(next_batch).detach()
        target = q_vals.clone()

        for i in range(self.batch_size):
            target[i][action_batch[i]] = reward_batch[i] + \
                self.gamma * torch.max(next_q_vals[i]) * (1 - done_batch[i])

        self.optimizer.zero_grad()
        loss = self.loss_fn(q_vals, target)
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    def save(self, filename="dqn_model.pth"):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename="dqn_model.pth"):
        self.model.load_state_dict(torch.load(filename))
        self.model.eval()
