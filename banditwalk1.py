import gym
import numpy as np

# Define the transition dictionary (MDP)
P = {
    0: {  # State 0
        0: (1.0, 0, 0.0, True),  # Action 0 -> Prob=1.0, Next State=0, Reward=0, Terminal=True
        1: (1.0, 0, 0.0, True)   # Action 1 -> Prob=1.0, Next State=0, Reward=0, Terminal=True
    },
    1: {  # State 1
        0: (1.0, 0, 0.0, True),  # Action 0 -> Goes back to state 0
        1: (1.0, 2, 1.0, True)   # Action 1 -> Goes to state 2, gets reward 1
    },
    2: {  # State 2
        0: (1.0, 2, 0.0, True),  # Stays at 2
        1: (1.0, 2, 0.0, True)   # Stays at 2
    }
}

# Define state and actions
state = 1  # Start from state 1
actions = [0, 1]  # Two possible actions (0 and 1)

def step(state, action):
    prob, next_state, reward, done = P[state][action]  # Get transition details
    return next_state, reward, done

# Reset environment
state = 1  # Start at state 1
done = False

# Run until terminal state is reached
while not done:
    action = np.random.choice(actions)  # Select a random action (0 or 1)
    next_state, reward, done = step(state, action)  # Take a step
    print(f"State: {state} → Action: {action} → Next State: {next_state}, Reward: {reward}")
    state = next_state  # Update current state

print("Episode finished!")
