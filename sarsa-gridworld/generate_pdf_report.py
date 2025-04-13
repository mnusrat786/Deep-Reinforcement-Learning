from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from utils import plot_policy, plot_heatmap, plot_visits

def generate_pdf_report(agent, total_rewards, trajectory, algo, episodes):
    with PdfPages("RL_Report.pdf") as pdf:
        # Title Page
        fig1 = plt.figure(figsize=(8, 6))
        fig1.clf()
        fig1.text(0.5, 0.8, "RL Simulator Report", ha='center', fontsize=20)
        fig1.text(0.5, 0.6, f"Algorithm: {algo}", ha='center', fontsize=14)
        fig1.text(0.5, 0.55, f"Episodes Trained: {episodes}", ha='center', fontsize=12)
        pdf.savefig(fig1)
        plt.close(fig1)

        # Policy Grid
        fig2 = plot_policy(agent.Q)
        pdf.savefig(fig2)
        plt.close(fig2)

        # Q-Value Heatmap
        fig3 = plot_heatmap(agent.Q)
        pdf.savefig(fig3)
        plt.close(fig3)

        # State Visit Frequency
        fig4 = plot_visits(agent.visits)
        pdf.savefig(fig4)
        plt.close(fig4)

        # Reward over Time
        fig5 = plt.figure(figsize=(8, 3))
        plt.plot(total_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.title("Reward vs Episode")
        pdf.savefig(fig5)
        plt.close(fig5)

        # Last Episode Path
        fig6 = plt.figure()
        fig6.clf()
        path_str = " → ".join([chr(65 + s[0]*4 + s[1]) for s in trajectory])
        fig6.text(0.5, 0.7, "Last Episode Path", ha='center', fontsize=16)
        fig6.text(0.5, 0.5, path_str, ha='center', fontsize=12)
        pdf.savefig(fig6)
        plt.close(fig6)

    print("✅ PDF Report saved as: RL_Report.pdf")
