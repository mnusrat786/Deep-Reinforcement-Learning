import random
from collections import defaultdict

class RLAgent:
    def __init__(self, actions, alpha=0.1, gamma=1.0, epsilon=0.1, method='sarsa'):
        self.Q = defaultdict(lambda: {a: 0.0 for a in actions})
        self.visits = defaultdict(int)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = actions
        self.method = method

    def choose_action(self, state):
        self.visits[state] += 1
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return max(self.Q[state], key=self.Q[state].get)

    def update(self, s, a, r, s_, a_):
        predict = self.Q[s][a]
        if self.method == 'sarsa':
            target = r + self.gamma * self.Q[s_][a_]
        else:
            target = r + self.gamma * max(self.Q[s_].values())
        self.Q[s][a] += self.alpha * (target - predict)