import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    },
)
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import numpy as np
from datetime import datetime
import seaborn as sns


def load_df(source):
    df = pd.read_csv(source)
    df['entropy'] = df['ability_sd'].map(
        lambda sd: 0.5 * np.log(2 * 3.1415928 * sd * sd) + 0.5,
    )
    return df


# Load the data
adaptive = load_df('output/KnowledgeTrial_adaptive.csv')
static = load_df('output/KnowledgeTrial_static.csv')

# Create a figure with subplots for different visualization approaches
fig, ax = plt.subplots(1, 1, figsize=(20, 16))

# 1. Individual lines for each participant (top-left)
participants = sorted(adaptive['participant_id'].unique())

# Create a color palette with enough colors for all participants
colors_adaptive = plt.cm.Reds(np.linspace(0, 1, len(participants)))
colors_static = plt.cm.Blues(np.linspace(0, 1, len(participants)))

for i, participant in enumerate(participants):
    p_data_adaptive = adaptive[
        adaptive['participant_id'] == participant].sort_values(
        'id',
    )
    p_data_static = static[static['participant_id'] == participant].sort_values(
        'id',
    )

    ax.plot(
        range(1, len(p_data_adaptive) + 1), p_data_adaptive['entropy'],
        alpha=0.7, linewidth=1, color=colors_adaptive[i],
        label=f'P{participant}',
    )
    ax.plot(
        range(1, len(p_data_static) + 1), p_data_static['entropy'],
        alpha=0.7, linewidth=1, color=colors_static[i], label=f'P{participant}',
    )

ax.set_xlabel('Trial Number')
ax.set_ylabel('Ability Mean')
ax.set_title('Individual Participant Trajectories')
ax.grid(True, alpha=0.3)
# Don't show legend for individual lines as it would be too cluttered

plt.show()

# Create a figure with subplots for different visualization approaches
fig, ax = plt.subplots(figsize=(4.8, 3.2))

n = adaptive.groupby("participant_id")["id"].count()
samples_adaptive = np.load("output/irt_samples_adaptive.npz")
samples_static = np.load("output/irt_samples_static.npz")
theta_adaptive = samples_adaptive['theta'].mean(axis=0)
theta_static = samples_static['theta'].mean(axis=0)

# ax.hist(n, bins=np.arange(0, n.max()+1))
scatter = ax.scatter(theta_static, theta_adaptive, c=n, cmap=plt.cm.Reds)
ax.set_xlabel(r"Final $E(\theta_i)$ (static)")
ax.set_ylabel(r"Final $E(\theta_i)$ (adaptive)")

r2 = np.corrcoef(theta_static, theta_adaptive)[0, 1] ** 2
ax.text(
    0.075, 0.95, f"$R^2={r2:.2f}$", transform=ax.transAxes, ha="left", va="top",
)
ax.text(
    0.05, 0.85, f"$\\langle n_i \\rangle={n.mean():.1f}$", transform=ax.transAxes, ha="left", va="top",
)

# Add colorbar with legend
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(
    r'$n_{i}^{trials}$', rotation=270, labelpad=15,
)  # 'n' as the colorbar label
plt.show()
plt.savefig("output/theta_comparison.pdf", bbox_inches="tight")
