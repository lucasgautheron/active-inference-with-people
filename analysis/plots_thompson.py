import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm

import scipy

matplotlib.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    },
)
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}"
)

import seaborn as sns

import pandas as pd
import numpy as np
from scipy.stats import entropy
import json
import logging

# Set up logger (adjust as needed for your logging setup)
logger = logging.getLogger(__name__)


def thompson_sampling(df):
    """
    Thompson sampling function adapted for pandas DataFrame input.

    Args:
        df (pd.DataFrame): DataFrame containing trial data

    Returns:
        int: ID of the best node selected by Thompson sampling
    """
    n_samples = 10000

    # Convert string boolean values to actual booleans
    df = df.copy()
    for col in ["y", "z"]:
        df[col] = df[col].map(
            {
                "True": True,
                "False": False,
                True: True,
                False: False,
            }
        )

    # Draw Phi (global parameter)
    alpha, beta = 1, 1

    # Get unique participants and their z values
    participants = df.groupby("participant_id")["z"].first()

    for z_val in participants:
        if z_val == True:
            alpha += 1
        elif z_val == False:
            beta += 1

    Phi = np.random.beta(alpha, beta, n_samples)

    node_ids = df["node_id"].unique()
    rewards = {}

    for node_id in node_ids:
        # Get trials for this node (equivalent to node.viable_trials)
        node_trials = df[df["node_id"] == node_id]

        alpha_node = np.ones(
            2
        )  # [alpha for z=0, alpha for z=1]
        beta_node = np.ones(
            2
        )  # [beta for z=0, beta for z=1]

        for _, trial in node_trials.iterrows():
            if pd.isna(trial["z"]):
                continue

            z = 1 if trial["z"] else 0
            if trial["y"] == True:
                alpha_node[z] += 1
            elif trial["y"] == False:
                beta_node[z] += 1

        # Sample phi for this node
        phi = np.random.beta(
            alpha_node[:, np.newaxis],
            beta_node[:, np.newaxis],
            (2, n_samples),
        )

        # Calculate expected information gain (reward)
        rewards[node_id] = entropy(
            [Phi, 1 - Phi], axis=0, base=2
        )

        for z in [False, True]:
            p_z = Phi if z else 1 - Phi
            for y in [False, True]:
                p_y_given_z = (
                    phi[z * 1] if y else 1 - phi[z * 1]
                )
                p_y = (
                    phi[1] * Phi + phi[0] * (1 - Phi)
                    if y
                    else (
                        (1 - phi[1]) * Phi
                        + (1 - phi[0]) * (1 - Phi)
                    )
                )

                rewards[node_id] += (
                    p_y_given_z
                    * p_z
                    * np.log2(p_y_given_z * p_z / p_y)
                )

        assert (rewards[node_id] > -1e-6).all()

    return rewards


# Load the data
full = pd.read_csv(
    "output/KnowledgeTrial_thompson_full.csv"
)
thompson_10 = pd.read_csv(
    "output/KnowledgeTrial_thompson_10.csv"
)
thompson_5 = pd.read_csv(
    "output/KnowledgeTrial_thompson_5.csv"
)

rewards_full = thompson_sampling(full)
rewards_thompson_10 = thompson_sampling(thompson_10)
rewards_thompson_5 = thompson_sampling(thompson_5)

mean_reward = {
    node: np.mean(rewards_full[node])
    for node in rewards_full.keys()
}
low_reward = {
    node: np.quantile(rewards_full[node], q=0.1 / 2)
    for node in rewards_full.keys()
}
high_reward = {
    node: np.quantile(rewards_full[node], q=1 - 0.1 / 2)
    for node in rewards_full.keys()
}
order = sorted(
    list(mean_reward.keys()),
    key=lambda node: mean_reward[node],
)

cmap = plt.get_cmap("Blues")
norm = plt.Normalize(
    vmin=0, vmax=np.max(list(mean_reward.values()))
)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Required for ScalarMappable

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    sharex=True,
    sharey=True,
    figsize=(3.2 * 1.25, 2.13333 * 1.25),
)
for i, node_id in enumerate(order):
    # Get item_id for coloring (assuming it's available in the data)
    item_id = (
        full[full["node_id"] == node_id]["item_id"].iloc[0]
        if len(full[full["node_id"] == node_id]) > 0
        else None
    )
    color = cmap(
        mean_reward[node_id]
        / np.max(list(mean_reward.values()))
    )
    sns.kdeplot(
        rewards_full[node_id],
        ax=axes[0],
        color=color,
        label=f"Node {node_id}",
    )
    sns.kdeplot(
        rewards_thompson_10[node_id],
        ax=axes[1],
        color=color,
        label=f"Node {node_id}",
    )
    sns.kdeplot(
        rewards_thompson_5[node_id],
        ax=axes[2],
        color=color,
        label=f"Node {node_id}",
    )

for i in range(3):
    axes[i].set_ylabel(r"")

axes[1].set_ylabel(r"$p(E[r_j|\phi_j,\Phi])$")
axes[2].set_xlabel(r"$E[r_j|\phi_j,\Phi]$")

axes[0].text(
    0.975,
    0.95,
    r"{\scriptsize Static design (100\% of trials)}",
    ha="right",
    va="top",
    transform=axes[0].transAxes,
)
axes[1].text(
    0.975,
    0.95,
    r"{\scriptsize Thompson sampling (67\% of trials)}",
    ha="right",
    va="top",
    transform=axes[1].transAxes,
)
axes[2].text(
    0.975,
    0.95,
    r"{\scriptsize Thompson sampling (33\% of trials)}",
    ha="right",
    va="top",
    transform=axes[2].transAxes,
)


cbar = plt.colorbar(
    sm, ax=axes
)  # Use the ScalarMappable here
cbar.set_label(
    r"$E[r_j]_{full}$",
    rotation=90,
    labelpad=15,
)
fig.savefig(
    "output/reward_distribution.pdf", bbox_inches="tight"
)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

nodes = list(mean_reward.keys())
counts = (thompson_10.groupby("node_id")["id"].count())[
    nodes
]
rewards = np.array(list(mean_reward.values()))
low_rewards = np.array(list(low_reward.values()))
high_rewards = np.array(list(high_reward.values()))
ax.scatter(rewards, counts)
ax.errorbar(
    rewards,
    counts,
    xerr=(rewards - low_rewards, high_rewards - rewards),
    ls="none",
    alpha=0.5,
)
ax.set_xlabel("$E[r_j]$ (Expected reward for $j$)")
ax.set_ylabel("Number of trials $n_j$")
plt.show()

unique_nodes = sorted(thompson_10["node_id"].unique())

# Initialize counters
counts = {node: 0 for node in unique_nodes}
cumulative_data = {node: [] for node in unique_nodes}

# Calculate cumulative counts for each row
for _, row in thompson_10.iterrows():
    counts[row["node_id"]] += 1

    # Record current count for all nodes
    for node in unique_nodes:
        cumulative_data[node].append(counts[node])

# Plot
fig, ax = plt.subplots(figsize=(3.2, 2.13333))

# Create a ScalarMappable for the colorbar
norm = plt.Normalize(
    vmin=0, vmax=np.max(list(mean_reward.values()))
)
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])  # Required for ScalarMappable

for node in unique_nodes:
    color = cmap(
        mean_reward[node]
        / np.max(list(mean_reward.values()))
    )
    ax.plot(cumulative_data[node], color=color)

ax.set_xlabel("Step $t$")
ax.set_ylabel(
    "Cumulative frequency\nof treatment $j$\nat step $t$"
)
cbar = plt.colorbar(
    sm, ax=ax
)  # Use the ScalarMappable here
cbar.set_label(
    r"$E[r_j]_{full}$",
    rotation=90,
    labelpad=15,
)
fig.savefig(
    "output/cumulative_node_frequency.pdf",
    bbox_inches="tight",
)

counts = {node: 0 for node in unique_nodes}
cumulative_data = {node: [] for node in unique_nodes}

# Calculate cumulative counts for each row
for _, row in thompson_5.iterrows():
    counts[row["node_id"]] += 1

    # Record current count for all nodes
    for node in unique_nodes:
        cumulative_data[node].append(counts[node])

# Plot
fig, ax = plt.subplots()
for node in unique_nodes:
    color = cmap(
        mean_reward[node]
        / np.max(list(mean_reward.values()))
    )
    ax.plot(cumulative_data[node], color=color)

ax.set_xlabel("Step $t$")
ax.set_ylabel(
    "Cumulative frequency of each treatment $j$ at step $t$"
)
plt.show()
