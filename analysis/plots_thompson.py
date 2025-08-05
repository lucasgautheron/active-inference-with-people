import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import textwrap

matplotlib.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    },
)
matplotlib.rcParams["text.latex.preamble"] = (
    r"\usepackage{amsmath}\linespread{1}"
)

import seaborn as sns

import pandas as pd
import numpy as np
from scipy.stats import entropy


def thompson_sampling(df):
    """
    Thompson sampling function adapted for pandas DataFrame input.

    Args:
        df (pd.DataFrame): DataFrame containing trial data

    Returns:
        int: ID of the best node selected by Thompson sampling
    """
    n_samples = 50000

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
random_10 = pd.read_csv(
    "output/KnowledgeTrial_thompson_10_random.csv"
)
thompson_5 = pd.read_csv(
    "output/KnowledgeTrial_thompson_5.csv"
)

rewards_full = thompson_sampling(full)
rewards_thompson_10 = thompson_sampling(thompson_10)
rewards_random_10 = thompson_sampling(random_10)
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
        rewards_random_10[node_id],
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
    r"{\scriptsize Greedy design (100\% of trials)}",
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
    r"{\scriptsize Even sampling (67\% of trials)}",
    ha="right",
    va="top",
    transform=axes[2].transAxes,
)


cbar = plt.colorbar(
    sm, ax=axes
)  # Use the ScalarMappable here
cbar.set_label(
    r"$E[r_j]_{greedy}$",
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


def cumulative_frequency(df, output):
    unique_nodes = sorted(df["node_id"].unique())

    # Initialize counters
    counts = {node: 0 for node in unique_nodes}
    cumulative_data = {node: [] for node in unique_nodes}

    # Calculate cumulative counts for each row
    for _, row in df.iterrows():
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
        r"$E[r_j]_{greedy}$",
        rotation=90,
        labelpad=15,
    )
    fig.savefig(
        output,
        bbox_inches="tight",
    )


cumulative_frequency(
    thompson_10, "output/cumulative_node_frequency.pdf"
)
cumulative_frequency(
    random_10, "output/cumulative_node_frequency_random.pdf"
)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))
mean_reward_thompson_10 = {
    node: np.mean(rewards_thompson_10[node])
    for node in rewards_thompson_10.keys()
}
mean_reward_random_10 = {
    node: np.mean(rewards_random_10[node])
    for node in rewards_random_10.keys()
}

nodes = rewards_full.keys()


def get_rank(r, nodes):
    v = np.array(list(r.values()))
    return [np.sum([r[node] <= v]) for node in nodes]


ranks_full = get_rank(mean_reward, nodes)
ranks_thompson_10 = get_rank(mean_reward_thompson_10, nodes)
ranks_random_10 = get_rank(mean_reward_random_10, nodes)

ax.plot([1, 15], [1, 15], color="black", zorder=0)
ax.scatter(
    ranks_full, ranks_thompson_10, label="Thompson sampling"
)
ax.scatter(
    ranks_full, ranks_random_10, label="Even sampling"
)
ax.set_xlabel("Treatment rank\nin the greedy setup")
ax.set_ylabel("Estimated treatment rank")

ax.set_xticks(
    [1, 5, 10, 15],
    ["1st\n(best)", "5th", "10th", "15th\n(worst)"],
)
ax.set_yticks(
    [1, 5, 10, 15],
    ["1st\n(best)", "5th", "10th", "15th\n(worst)"],
)

fig.legend(
    frameon=False,
    bbox_to_anchor=(0.5, 1.05),
    loc="upper center",
    ncol=2,
)
plt.savefig("output/ranks.pdf", bbox_inches="tight")


def plot_y_distributions_by_z(df, mean_reward):
    """
    Creates a 5x3 grid of plots showing beta distributions fitted to observed y values
    for each z condition (0,1) across all 15 node_ids, ordered by mean reward from highest to lowest.

    Args:
        df (pd.DataFrame): DataFrame containing the trial data with columns
                          ['node_id', 'y', 'z']
        mean_reward (dict): Dictionary mapping node_id to mean reward values
        figsize (tuple): Figure size (width, height)

    Returns:
        fig, axes: matplotlib figure and axes objects
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from scipy.stats import beta

    questions = (
        pd.read_csv("static/questions.csv")
        .set_index("id")["question"]
        .to_dict()
    )

    # Order nodes by mean reward (highest to lowest)
    ordered_nodes = sorted(
        mean_reward.keys(),
        key=lambda node: mean_reward[node],
        reverse=True,
    )

    # Create 5x3 subplot grid
    fig, axes = plt.subplots(
        3, 5, figsize=(6.4, 4.8), sharex=True, sharey=True
    )
    axes = axes.flatten()  # Flatten for easier indexing

    # Convert string boolean values to actual booleans if needed
    df_clean = df.copy()
    for col in ["y", "z"]:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].map(
                {
                    "True": True,
                    "False": False,
                    True: True,
                    False: False,
                    1: True,
                    0: False,
                }
            )

    # Plot each node
    for i, node_id in enumerate(ordered_nodes):
        ax = axes[i]

        # Filter data for this node
        node_data = df_clean[df_clean["node_id"] == node_id]

        if len(node_data) == 0:
            ax.text(
                0.5,
                0.5,
                f"Node {node_id}\n(No data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        # Separate data by z value
        z0_data = node_data[node_data["z"] == False]
        z1_data = node_data[node_data["z"] == True]

        # Create x values for plotting beta distributions
        x = np.linspace(0, 1, 100)

        # Plot beta distribution for z=0
        if len(z0_data) > 0:
            successes_z0 = z0_data[
                "y"
            ].sum()  # Count of True values
            failures_z0 = (
                len(z0_data) - successes_z0
            )  # Count of False values
            alpha_z0 = (
                successes_z0 + 1
            )  # Beta prior with alpha=1, beta=1
            beta_z0 = failures_z0 + 1

            y_z0 = beta.pdf(x, alpha_z0, beta_z0)
            ax.plot(
                x,
                y_z0,
                color="#377eb8",
                linewidth=2,
                label=(
                    "High-scool or less ($z=0$)"
                    if i == 0
                    else None
                ),
            )
            ax.fill_between(
                x, y_z0, alpha=0.3, color="#377eb8"
            )

        # Plot beta distribution for z=1
        if len(z1_data) > 0:
            successes_z1 = z1_data[
                "y"
            ].sum()  # Count of True values
            failures_z1 = (
                len(z1_data) - successes_z1
            )  # Count of False values
            alpha_z1 = (
                successes_z1 + 1
            )  # Beta prior with alpha=1, beta=1
            beta_z1 = failures_z1 + 1

            y_z1 = beta.pdf(x, alpha_z1, beta_z1)
            ax.plot(
                x,
                y_z1,
                color="#ff7f00",
                linewidth=2,
                label=(
                    "College ($z=1$)" if i == 0 else None
                ),
            )
            ax.fill_between(
                x, y_z1, alpha=0.3, color="#ff7f00"
            )

        # Customize subplot
        ax.set_title(
            f"$E[r_j]={mean_reward[node_id]:.2f}$",
        )
        question = "\n\\tiny ".join(
            textwrap.wrap(questions[node_id - 1], 20)
        )
        ax.text(
            0.025,
            0.975,
            f"\\tiny ``{question}''",
            transform=ax.transAxes,
            ha="left",
            va="top",
            linespacing=1
        )
        ax.set_ylim(0, 50)
        ax.set_xlim(0, 1)

    fig.legend(
        loc="upper right",
        fontsize=8,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(1, 1.1),
    )

    plt.tight_layout()
    plt.subplots_adjust(
        top=0.985, bottom=0.015, left=0.015, right=0.985
    )

    plt.savefig(
        "output/posteriors.pdf", bbox_inches="tight"
    )

    return fig, axes


plot_y_distributions_by_z(full, mean_reward)
