#  Interactive SARSA & Q-Learning GridWorld Simulator

This is a fully interactive Streamlit-based simulator that lets you visualize how Reinforcement Learning agents (SARSA and Q-Learning) learn to navigate a grid environment avoiding holes and reaching a goal.



---

##  Features

-  SARSA and Q-Learning toggle
-  Total Reward vs Episode graph
-  Epsilon-greedy exploration control
-  Policy grid visualization (with arrows & emojis)
-  Q-value heatmap and State visit heatmap
-  One-click PDF report export
-  Last episode agent path display
-  Built using: `streamlit`, `matplotlib`, `numpy`

---

## How to Run

```bash
git clone https://github.com/your-username/rl-gridworld-simulator.git
cd rl-gridworld-simulator
python -m venv venv
source venv/bin/activate     # or venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

