from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import os

def generate_pdf_report(agent, rewards, trajectory, algo, episodes, gif_path="agent_walk.gif"):
    doc = SimpleDocTemplate("RL_Report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Reinforcement Learning Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Algorithm: {algo}", styles['Normal']))
    elements.append(Paragraph(f"Episodes Trained: {episodes}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Reward plot
    fig, ax = plt.subplots()
    ax.plot(rewards)
    ax.set_title("Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    plt.grid(True)
    fig.savefig("reward_plot.png")
    plt.close(fig)

    elements.append(Paragraph("Reward Curve:", styles['Heading2']))
    elements.append(Image("reward_plot.png", width=400, height=200))
    elements.append(Spacer(1, 12))

    # Agent trajectory
    elements.append(Paragraph("Agent Trajectory (Last Episode):", styles['Heading2']))
    traj_text = " → ".join([chr(65 + s[0]*4 + s[1]) for s in trajectory])
    elements.append(Paragraph(traj_text, styles['Code']))
    elements.append(Spacer(1, 12))

    # GIF snapshot (first frame)
    if os.path.exists(gif_path):
        elements.append(Paragraph("Agent Walk (GIF snapshot):", styles['Heading2']))
        elements.append(Image(gif_path, width=200, height=200))

    doc.build(elements)

    # Clean temp
    if os.path.exists("reward_plot.png"):
        os.remove("reward_plot.png")
