import streamlit as st
import matplotlib.pyplot as plt
from environment import GridWorld
from agent import RLAgent
from utils import (
    plot_policy,
    plot_heatmap,
    plot_visits,
    save_agent_walk_gif
)
from generate_pdf_report import generate_pdf_report

st.set_page_config(layout="wide")
st.title("🎮 Interactive SARSA / Q-Learning GridWorld Simulator")

# Sidebar: hyperparameters
algo = st.sidebar.radio("Select Algorithm", ["SARSA", "Q-Learning"])
alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.0, 1.0, 1.0)
initial_epsilon = st.sidebar.slider("Initial Exploration Rate (ε)", 0.0, 1.0, 0.1)
episodes = st.sidebar.number_input("Episodes", 10, 5000, 100)

# Initialize environment and agent
env = GridWorld()
agent = RLAgent(env.actions, alpha, gamma, initial_epsilon, method=algo.lower())

# Training
total_rewards = []
all_trajectories = []
best_idx = 0
worst_idx = 0

for ep in range(episodes):
    state = env.reset()
    action = agent.choose_action(state)
    done = False
    total_reward = 0
    trajectory = [state]
    agent.epsilon = max(0.01, initial_epsilon * (0.995 ** ep))

    while not done:
        next_state, reward, done = env.step(action)
        next_action = agent.choose_action(next_state)
        agent.update(state, action, reward, next_state, next_action)
        state, action = next_state, next_action
        trajectory.append(state)
        total_reward += reward

    total_rewards.append(total_reward)
    all_trajectories.append(trajectory)

    if total_reward > total_rewards[best_idx]:
        best_idx = ep
    if total_reward < total_rewards[worst_idx]:
        worst_idx = ep

st.success(f"Training completed over {episodes} episodes using **{algo}**.")

# Grid visualizations
st.pyplot(plot_policy(agent.Q))
st.pyplot(plot_heatmap(agent.Q))
st.pyplot(plot_visits(agent.visits))

# Reward graph
st.subheader("Total Reward per Episode")
fig = plt.figure(figsize=(8, 3))
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
st.pyplot(fig)

# Agent path
st.subheader("🚶 Agent Path in Last Episode")
st.write(" → ".join([chr(65 + s[0]*4 + s[1]) for s in all_trajectories[-1]]))

# Q-value viewer
st.subheader(" Inspect Q-Values")
selected_state = st.selectbox("Choose a state (row, col)", [(i, j) for i in range(4) for j in range(4)])
if selected_state in agent.Q:
    st.json(agent.Q[selected_state])
else:
    st.info("No Q-values learned for this state.")

# Walk animation dropdown
st.subheader("🎞️ Agent Walk (GIF)")
episode_option = st.selectbox("Select Episode to Animate", ["Last Episode", "Best Episode", "Worst Episode"])

if episode_option == "Last Episode":
    ep_idx = episodes - 1
elif episode_option == "Best Episode":
    ep_idx = best_idx
else:
    ep_idx = worst_idx

if st.button("Generate Walk Animation"):
    save_agent_walk_gif(all_trajectories[ep_idx], episode=ep_idx + 1)
    st.image("agent_walk.gif", caption=f"Agent Movement - {episode_option}")

# PDF export
st.subheader(" Export Report")
if st.button("Generate PDF Report"):
    generate_pdf_report(
        agent=agent,
        rewards=total_rewards,
        trajectory=all_trajectories[-1],
        algo=algo,
        episodes=episodes,
        gif_path="agent_walk.gif"
    )
    st.success(" RL_Report.pdf saved in your folder")
