import gym
import numpy as np
from gym import spaces

class BanditWalkEnv(gym.Env):
    """
    Custom Bandit Walk MDP environment.
    """
    def __init__(self):
        super(BanditWalkEnv, self).__init__()

        # Define state space (three states: 0, 1, 2)
        self.state_space = [0, 1, 2]
        self.observation_space = spaces.Discrete(len(self.state_space))

        # Define action space (two actions: 0, 1)
        self.action_space = spaces.Discrete(2)

        # Define transition probabilities and rewards
        self.P = {
            0: {
                0: [(1.0, 0, 0.0, True)],  # (prob, next_state, reward, done)
                1: [(1.0, 0, 0.0, True)]
            },
            1: {
                0: [(1.0, 0, 0.0, True)],
                1: [(1.0, 2, 1.0, True)]
            },
            2: {
                0: [(1.0, 2, 0.0, True)],
                1: [(1.0, 2, 0.0, True)]
            }
        }

        self.state = 0  # Initialize the environment in state 0

    def step(self, action):
        """
        Execute one step in the environment.
        """
        transitions = self.P[self.state][action]
        prob, next_state, reward, done = transitions[0]

        self.state = next_state
        return self.state, reward, done, {}

    def reset(self):
        """
        Reset environment to initial state.
        """
        self.state = 0
        return self.state

    def render(self, mode='human'):
        """
        Print current state.
        """
        print(f"Current state: {self.state}")

# Test the environment
env = BanditWalkEnv()
state = env.reset()
print(f"Initial state: {state}")

for _ in range(2):  # Simulate 2 episodes
    action = np.random.choice([0, 1])  # Randomly choose action
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Next state: {next_state}, Reward: {reward}, Done: {done}")
    env.render()
    if done:
        env.reset()
