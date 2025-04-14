import matplotlib.pyplot as plt
import numpy as np
import imageio
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def draw_grid_frame(path, grid_size=(4, 4), holes=[(1, 1), (1, 3), (3, 0)], goal=(3, 3), episode=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xticks(np.arange(grid_size[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(grid_size[0]+1)-0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xticks([])
    ax.set_yticks([])

    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            label = chr(65 + i * grid_size[1] + j)
            color = "white"
            if (i, j) in holes:
                color = "#ffcccc"
            elif (i, j) == goal:
                color = "#ccffcc"
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color))
            ax.text(j, i, label, ha='center', va='center', fontsize=12)

    for i, (x, y) in enumerate(path):
        ax.add_patch(plt.Circle((y, x), 0.3, color='blue', alpha=0.3 + 0.5 * (i/len(path))))

    if path:
        x, y = path[-1]
        ax.add_patch(plt.Circle((y, x), 0.3, color='red'))

    ax.set_xlim(-0.5, grid_size[1]-0.5)
    ax.set_ylim(-0.5, grid_size[0]-0.5)
    ax.invert_yaxis()

    if episode is not None:
        ax.set_title(f"Episode {episode}", fontsize=14)

    return fig

def save_agent_walk_gif(trajectory, filename="agent_walk.gif", episode=None, loop=True):
    frames = []
    for i in range(1, len(trajectory) + 1):
        fig = draw_grid_frame(trajectory[:i], episode=episode)
        canvas = FigureCanvas(fig)
        canvas.draw()
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(image)
        plt.close(fig)
    
    imageio.mimsave(filename, frames, duration=0.5, loop=0 if loop else 1)




def plot_policy(Q, grid_size=4):
    fig, ax = plt.subplots(figsize=(6, 6))
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    labels = np.array([chr(65 + i * grid_size + j) for i in range(grid_size) for j in range(grid_size)])
    labels = labels.reshape((grid_size, grid_size))
    holes = [(1, 1), (1, 3), (3, 0)]
    goal = (3, 3)
    start = (0, 0)

    cell_text = []
    cell_colors = []

    for i in range(grid_size):
        row_text = []
        row_color = []
        for j in range(grid_size):
            state = (i, j)
            label = labels[i, j]

            if state in holes:
                row_text.append(f"❌\n{label}")
                row_color.append("#ffcccc")
            elif state == goal:
                row_text.append(f"✅\n{label}")
                row_color.append("#ccffcc")
            elif state == start:
                row_text.append(f"🟦\n{label}")
                row_color.append("#cce5ff")
            elif state in Q:
                best_action = max(Q[state], key=Q[state].get)
                row_text.append(f"{arrows[best_action]}\n{label}")
                row_color.append("white")
            else:
                row_text.append(label)
                row_color.append("white")

        cell_text.append(row_text)
        cell_colors.append(row_color)

    table = ax.table(cellText=cell_text,
                     cellColours=cell_colors,
                     loc='center',
                     cellLoc='center',
                     colWidths=[0.2]*grid_size)

    table.scale(1, 2)
    ax.axis('off')
    plt.tight_layout()
    return fig

def plot_heatmap(Q):
    fig, ax = plt.subplots()
    values = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            state = (i, j)
            if state in Q:
                values[i][j] = max(Q[state].values())
    c = ax.imshow(values, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(c)
    ax.set_title("Q-Value Heatmap (Best Actions)")
    return fig

def plot_visits(visits):
    fig, ax = plt.subplots()
    counts = np.zeros((4, 4))
    for (i, j), count in visits.items():
        counts[i][j] = count
    c = ax.imshow(counts, cmap='YlGn', interpolation='nearest')
    plt.colorbar(c)
    ax.set_title("State Visit Frequency")
    return fig

def plot_dqn_qvalues(agent_model, actions):
    q_grid = np.zeros((4, 4))
    with torch.no_grad():
        for i in range(4):
            for j in range(4):
                input_tensor = torch.FloatTensor([i, j])
                q_vals = agent_model(input_tensor)
                best_action_val = torch.max(q_vals).item()
                q_grid[i, j] = best_action_val

    fig, ax = plt.subplots()
    im = ax.imshow(q_grid, cmap="coolwarm")
    ax.set_title("DQN Q-Value Heatmap (Best Actions)")
    plt.colorbar(im)
    return fig
