import streamlit as st
import matplotlib.pyplot as plt
from environment import GridWorld
from agent import RLAgent
from dqn_agent import DQNAgent
from utils import (
    plot_policy,
    plot_heatmap,
    plot_visits,
    save_agent_walk_gif,
    plot_dqn_qvalues
)
from generate_pdf_report import generate_pdf_report

st.set_page_config(layout="wide")
st.title("🎮 RL GridWorld: SARSA / Q-Learning / DQN")

# Sidebar
algo = st.sidebar.radio("Select Algorithm", ["SARSA", "Q-Learning", "DQN"])
alpha = st.sidebar.slider("Learning Rate (α)", 0.001, 1.0, 0.1)
gamma = st.sidebar.slider("Discount Factor (γ)", 0.0, 1.0, 0.99)
initial_epsilon = st.sidebar.slider("Initial Exploration Rate (ε)", 0.0, 1.0, 0.1)
episodes = st.sidebar.number_input("Episodes", 10, 5000, 300)

env = GridWorld()

# Agent selection
if algo == "DQN":
    agent = DQNAgent(env.actions, alpha=alpha, gamma=gamma, epsilon=initial_epsilon)

    st.sidebar.subheader("💾 DQN Model Options")
    if st.sidebar.button("Save Model"):
        agent.save()
        st.sidebar.success("Model saved as dqn_model.pth")

    if st.sidebar.button("Load Model"):
        try:
            agent.load()
            st.sidebar.success("Model loaded from dqn_model.pth")
        except:
            st.sidebar.error("Model file not found or invalid.")
else:
    agent = RLAgent(env.actions, alpha, gamma, initial_epsilon, method=algo.lower())

# Training
total_rewards = []
all_trajectories = []
best_idx = 0
worst_idx = 0

for ep in range(episodes):
    state = env.reset()
    action = agent.act(state) if algo == "DQN" else agent.choose_action(state)
    done = False
    total_reward = 0
    trajectory = [state]

    while not done:
        next_state, reward, done = env.step(action)

        if algo == "DQN":
            next_action = agent.act(next_state)
            agent.remember(state, action, reward, next_state, done)
            agent.train_step()
        else:
            next_action = agent.choose_action(next_state)
            agent.update(state, action, reward, next_state, next_action)

        state, action = next_state, next_action
        trajectory.append(state)
        total_reward += reward

    total_rewards.append(total_reward)
    all_trajectories.append(trajectory)

    if env.state == env.goal and total_reward > total_rewards[best_idx]:
        best_idx = ep

    if total_reward < total_rewards[worst_idx]:
        worst_idx = ep

st.success(f"✅ Training completed over {episodes} episodes using **{algo}**.")

# Visualizations
if algo == "DQN":
    st.pyplot(plot_dqn_qvalues(agent.model, env.actions))
else:
    st.pyplot(plot_policy(agent.Q))
    st.pyplot(plot_heatmap(agent.Q))
    st.pyplot(plot_visits(agent.visits))

# Reward Graph
st.subheader("📈 Total Reward per Episode")
fig = plt.figure(figsize=(8, 3))
plt.plot(total_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
st.pyplot(fig)

# Trajectory
st.subheader("🚶 Agent Path in Last Episode")
trajectory_str = " → ".join(dict.fromkeys([chr(65 + s[0]*4 + s[1]) for s in all_trajectories[-1]]))
st.code(trajectory_str, language="markdown")

# Q-table viewer
if algo != "DQN":
    st.subheader("🧠 Inspect Q-Values")
    selected_state = st.selectbox("Choose a state (row, col)", [(i, j) for i in range(4) for j in range(4)])
    if selected_state in agent.Q:
        st.json(agent.Q[selected_state])
    else:
        st.info("No Q-values learned yet for this state.")

# 🎞️ Walk animation + Loop + Download
st.subheader("🎞️ Agent Walk Animation")

episode_option = st.selectbox("Episode to Animate", ["Last Episode", "Best Episode", "Worst Episode"])
loop_forever = st.checkbox("🔁 Loop animation infinitely", value=True)
ep_idx = {"Last Episode": episodes-1, "Best Episode": best_idx, "Worst Episode": worst_idx}[episode_option]

if st.button("Generate Walk Animation"):
    save_agent_walk_gif(all_trajectories[ep_idx], episode=ep_idx + 1, loop=loop_forever)
    st.image("agent_walk.gif", caption=f"Agent Movement - {episode_option}")
    with open("agent_walk.gif", "rb") as f:
        st.download_button("⬇️ Download GIF", f, file_name="agent_walk.gif", mime="image/gif")

# 📄 PDF export
st.subheader("📄 Export Report")
if st.button("Generate PDF Report"):
    generate_pdf_report(agent, total_rewards, all_trajectories[-1], algo, episodes, gif_path="agent_walk.gif")
    st.success("✅ RL_Report.pdf saved in your folder")
