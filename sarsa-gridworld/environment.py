import numpy as np

class GridWorld:
    def __init__(self):
        self.size = 4
        self.state = (0, 0)
        self.goal = (3, 3)
        self.holes = [(1, 1), (1, 3), (3, 0)]
        self.actions = ['up', 'down', 'left', 'right']
        self.reset()

    def reset(self):
        self.state = (0, 0)
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up': x = max(0, x - 1)
        elif action == 'down': x = min(self.size - 1, x + 1)
        elif action == 'left': y = max(0, y - 1)
        elif action == 'right': y = min(self.size - 1, y + 1)
        self.state = (x, y)

        if self.state in self.holes:
            return self.state, -50, True
        elif self.state == self.goal:
            return self.state, 50, True
        else:
            return self.state, -1, False