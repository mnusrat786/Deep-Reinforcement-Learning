import matplotlib.pyplot as plt
import numpy as np

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
