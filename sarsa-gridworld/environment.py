import numpy as np

class GridWorld:
    def __init__(self, max_steps=50):
        self.size = 4
        self.start = (0, 0)
        self.goal = (3, 3)
        self.holes = [(1, 1), (1, 3), (3, 0)]
        self.actions = ['up', 'down', 'left', 'right']
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.state = self.start
        self.steps = 0
        return self.state

    def step(self, action):
        x, y = self.state
        self.steps += 1

        if action == 'up': x = max(0, x - 1)
        elif action == 'down': x = min(self.size - 1, x + 1)
        elif action == 'left': y = max(0, y - 1)
        elif action == 'right': y = min(self.size - 1, y + 1)

        self.state = (x, y)

        if self.state in self.holes:
            return self.state, -50, True
        elif self.state == self.goal:
            return self.state, 50, True
        elif self.steps >= self.max_steps:
            return self.state, -10, True  # episode timeout
        else:
            return self.state, -1, False

