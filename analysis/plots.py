import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import scipy

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

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

n = adaptive.groupby("participant_id")["id"].count()
samples_adaptive = np.load("output/irt_samples_adaptive.npz")
samples_static = np.load("output/irt_samples_static.npz")
theta_adaptive = samples_adaptive['theta'].mean(axis=0)
theta_static = samples_static['theta'].mean(axis=0)

difficulty_static = samples_static["d"].mean(axis=0)

scatter = ax.scatter(theta_static, theta_adaptive, c=n, cmap=plt.cm.Blues, s=4)
ax.set_xlabel(r"Posterior $\theta_i$ (static)")
ax.set_ylabel(r"Posterior $\theta_i$ (adaptive)")

r2 = np.corrcoef(theta_static, theta_adaptive)[0, 1] ** 2
ax.text(
    0.075, 0.95, f"$R^2={r2:.2f}$", transform=ax.transAxes, ha="left", va="top",
)
ax.text(
    0.05, 0.85, f"$\\langle n_i \\rangle={n.mean():.1f}$",
    transform=ax.transAxes, ha="left", va="top",
)

# Add colorbar with legend
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(
    r'$n_{i}^{trials}$', rotation=270, labelpad=15,
)  # 'n' as the colorbar label
plt.show()
plt.savefig("output/theta_comparison.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(3.2, 2.13333))
n = adaptive.groupby("node_id")["participant_id"].count()
n_static = static.groupby("node_id")["participant_id"].count()
ax.scatter(difficulty_static, n, label="Adaptive")
ax.scatter(difficulty_static, n_static, label="Static")
ax2 = ax.twinx()
sns.kdeplot(
    x=theta_static, label=r"$\theta_i$ distribution (static)", ax=ax2,
    hue_norm=(0, 200),
    bw_adjust=0.8, color='#4daf4a',
)
ax.set_xlabel("Challenge difficulty")
ax.set_ylabel("Number of trials")
ax.set_ylim(0, None)
fig.legend(
    ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.15), frameon=False,
)
plt.savefig("output/node_frequency.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(3.2, 2.13333))
n = adaptive.groupby("participant_id")["id"].count()
sd_adaptive = samples_adaptive["theta"].std(axis=0)
sd_static = samples_static["theta"].std(axis=0)
sd = samples_static["sigma_theta"]


def sd_to_entropy(sd):
    return 0.5 * np.log(2 * np.pi * sd * sd) + 0.5


entropy_adaptive = sd_to_entropy(sd_adaptive)
entropy_static = sd_to_entropy(sd_static)
prior_entropy = sd_to_entropy(sd).mean(axis=0)

# Original violin plots
ax.violinplot(
    [entropy_adaptive, entropy_static], positions=[1, 2],
)
# Prior entropy (position 0)
ax.plot([-0.125, 0.125], [prior_entropy] * 2, color="black")
ax.plot(
    [+0.125, ax.get_xlim()[1] - (-0.125 - ax.get_xlim()[0])],
    [prior_entropy] * 2, color="black", lw=0.5, ls="dashed",
)
ax.scatter([0], [prior_entropy], color="black", s=5)

# Add mean value text for the prior
ax.text(
    0, prior_entropy + 0.02, f'{prior_entropy:.2f}', ha='center', va='bottom',
)

# Adaptive posterior (position 1)
mean_val_adaptive = entropy_adaptive.mean()
ax.scatter([1], [mean_val_adaptive], color="black", s=5)
ax.plot(
    [-0.125 + 1, 0.125 + 1], [mean_val_adaptive] * 2,
    color="black",
)

# Add mean value text for adaptive
ax.text(
    1, mean_val_adaptive + 0.02, f'{mean_val_adaptive:.2f}', ha='center',
    va='bottom',
)

# Static posterior (position 2)
mean_val_static = entropy_static.mean()
ax.scatter([2], [mean_val_static], color="black", s=5)
ax.plot(
    [-0.125 + 2, 0.125 + 2], [mean_val_static] * 2,
    color="black",
)

# Add mean value text for static
ax.text(
    2, mean_val_static + 0.02, f'{mean_val_static:.2f}', ha='center',
    va='bottom',
)

# Add arrows showing information gain
# Arrow from prior to adaptive
adaptive_info_gain = prior_entropy - mean_val_adaptive
ax.annotate(
    '', xy=(0.75, mean_val_adaptive), xytext=(0.75, prior_entropy),
    arrowprops=dict(arrowstyle='->', lw=0.5),
)

# Information gain label for adaptive
ax.text(
    0.65, (prior_entropy + mean_val_adaptive) / 2,
    f'IG = {adaptive_info_gain:.2f}',
    ha='center', va='center', fontsize=8,
    rotation=90,
)

# Arrow from prior to static
static_info_gain = prior_entropy - mean_val_static
ax.annotate(
    '', xy=(1.75, mean_val_static), xytext=(1.75, prior_entropy),
    arrowprops=dict(arrowstyle='->', lw=0.5),
)

# Information gain label for static
ax.text(
    1.65, (prior_entropy + mean_val_static) / 2,
    f'IG = {static_info_gain:.2f}',
    ha='center', va='center', fontsize=8,
    rotation=90,
)

ax.set_xticks([0, 1, 2], ["No data (prior)", "Adaptive", "Static"])
ax.set_ylabel(r"$H(\theta_i)$ distribution")
ax.set_ylim(None, 2.0)

plt.savefig("output/entropy.pdf", bbox_inches="tight")
