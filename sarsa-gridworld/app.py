import streamlit as st
from environment import GridWorld
from agent import RLAgent
from utils import plot_policy, plot_heatmap, plot_visits
from generate_pdf_report import generate_pdf_report
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Interactive SARSA / Q-Learning GridWorld")

# Sidebar Settings
algo = st.sidebar.radio("Select Algorithm", ["SARSA", "Q-Learning"])
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.0, 1.0, 1.0)
initial_epsilon = st.sidebar.slider("Initial Exploration Rate (ε)", 0.0, 1.0, 0.1)
episodes = st.sidebar.number_input("Episodes", 10, 5000, 100)

# Environment and Agent
env = GridWorld()
agent = RLAgent(env.actions, alpha, gamma, initial_epsilon, method=algo.lower())

total_rewards = []
trajectory = []
min_epsilon = 0.01
decay_rate = 0.995

# Training Loop
for ep in range(episodes):
    state = env.reset()
    action = agent.choose_action(state)
    done = False
    total_reward = 0
    ep_trajectory = [state]
    agent.epsilon = max(min_epsilon, initial_epsilon * (decay_rate ** ep))

    while not done:
        next_state, reward, done = env.step(action)
        next_action = agent.choose_action(next_state)
        agent.update(state, action, reward, next_state, next_action)
        state, action = next_state, next_action
        ep_trajectory.append(state)
        total_reward += reward

    trajectory = ep_trajectory
    total_rewards.append(total_reward)

st.success(f" Training completed over {episodes} episodes using **{algo}**.")

# Visualizations
st.pyplot(plot_policy(agent.Q))
st.pyplot(plot_heatmap(agent.Q))
st.pyplot(plot_visits(agent.visits))

# Reward Graph
st.subheader(" Total Reward per Episode")
fig = plt.figure(figsize=(8, 3))
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
st.pyplot(fig)

# Trajectory
st.subheader("🚶 Agent Path in Last Episode")
st.write(" → ".join([chr(65 + s[0]*4 + s[1]) for s in trajectory]))

# Q-Value Viewer
st.subheader(" Inspect Q-Values")
selected_state = st.selectbox("State (row, col)", [(i, j) for i in range(4) for j in range(4)])
if selected_state in agent.Q:
    st.json(agent.Q[selected_state])
else:
    st.info("No Q-values learned yet for this state.")

# PDF Export
st.subheader(" Export")
if st.button("Generate PDF Report"):
    generate_pdf_report(agent, total_rewards, trajectory, algo, episodes)
    st.success("PDF saved as RL_Report.pdf in your folder ✅")
