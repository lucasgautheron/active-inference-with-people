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
    r"\usepackage{amsmath}\usepackage{amssymb}\linespread{1}"
)

import seaborn as sns

import pandas as pd
import numpy as np


def expected_free_energy(df):
    """
    Active inference function adapted for pandas DataFrame input.

    Args:
        df (pd.DataFrame): DataFrame containing trial data

    Returns:
        int: ID of the best node selected by Active inference
    """
    n_samples = 100000

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

    # Get unique participants and their z values
    participants = df.groupby("participant_id")["z"].first()

    alpha, beta = 1, 1
    for z_val in participants:
        if z_val == True:
            alpha += 1
        elif z_val == False:
            beta += 1

    Phi = np.random.beta(alpha, beta, n_samples)
    z = np.random.binomial(
        np.ones(n_samples, dtype=int), Phi
    )

    nodes_ids = np.arange(15) + 1

    rewards = dict()
    eig = dict()
    utility = dict()

    for node_id in nodes_ids:
        # Get trials for this node (equivalent to node.viable_trials)
        node_trials = df[df["node_id"] == node_id]

        alpha = np.ones(2)  # [alpha for z=0, alpha for z=1]
        beta = np.ones(2)  # [beta for z=0, beta for z=1]

        for _, trial in node_trials.iterrows():
            if pd.isna(trial["z"]):
                continue

            trial_z = 1 if trial["z"] else 0
            if trial["y"] == True:
                alpha[trial_z] += 1
            elif trial["y"] == False:
                beta[trial_z] += 1

        alpha = alpha[:, np.newaxis]
        beta = beta[:, np.newaxis]

        phi = np.random.beta(
            alpha,
            beta,
            (2, n_samples),
        )

        y = np.random.binomial(
            np.ones((2, n_samples), dtype=int),
            phi,
            size=(
                2,
                n_samples,
            ),
        )

        p_y_given_phi = phi * y + (1 - phi) * (1 - y)
        p_y = alpha / (alpha + beta) * y + beta / (
            alpha + beta
        ) * (1 - y)

        p_y_given_phi = p_y_given_phi[
            z, np.arange(n_samples)
        ]
        p_y = p_y[z, np.arange(n_samples)]
        EIG = np.mean(np.log(p_y_given_phi / p_y))

        gamma = 0.1
        p_z = z.mean()
        U = np.mean(
            np.log(
                p_z * np.exp(gamma * (y[1] - (1 - y[1])))
                + (1 - p_z)
                * np.exp(gamma * ((1 - y[0]) - y[0])),
            )
        )

        rewards[node_id] = EIG + U
        eig[node_id] = EIG
        utility[node_id] = U

        print(
            node_id,
            rewards[node_id],
            eig[node_id],
            utility[node_id],
        )

    return utility, eig, utility


# Load the data
full = pd.read_csv(
    "output/KnowledgeTrial_thompson_full.csv"
)
active_5 = pd.read_csv(
    "output/KnowledgeTrial_active_5.csv"
)
random_5 = pd.read_csv(
    "output/KnowledgeTrial_random_5.csv"
)
# random_5 = active_5
# thompson_5 = pd.read_csv(
#     "output/KnowledgeTrial_thompson_5.csv"
# )

rewards_full, eig_full, utility_full = expected_free_energy(
    full
)
rewards_active_5, eig_active_5, utility_active_5 = (
    expected_free_energy(active_5)
)
rewards_random_5, eig_random_5, utility_random_5 = (
    expected_free_energy(random_5)
)

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

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

nodes = list(mean_reward.keys())
# counts = (active_5.groupby("node_id")["id"].count())[nodes]
# rewards = np.array(list(mean_reward.values()))
# low_rewards = np.array(list(low_reward.values()))
# high_rewards = np.array(list(high_reward.values()))
# ax.scatter(rewards, counts)
# ax.errorbar(
#     rewards,
#     counts,
#     xerr=(rewards - low_rewards, high_rewards - rewards),
#     ls="none",
#     alpha=0.5,
# )
# ax.set_xlabel("$E[r_j]$ (Expected reward for $j$)")
# ax.set_ylabel("Number of trials $n_j$")
# plt.show()


def cumulative_frequency(df, output):
    cmap = plt.get_cmap("Blues")

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
        vmin=np.min(list(mean_reward.values())),
        vmax=np.max(list(mean_reward.values())),
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for ScalarMappable

    for node in unique_nodes:
        color = cmap(norm(mean_reward[node]))
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
    active_5, "output/cumulative_node_frequency.pdf"
)
cumulative_frequency(
    random_5, "output/cumulative_node_frequency_random.pdf"
)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))
mean_reward_active_5 = {
    node: np.mean(rewards_active_5[node])
    for node in rewards_active_5.keys()
}
mean_reward_random_5 = {
    node: np.mean(rewards_random_5[node])
    for node in rewards_random_5.keys()
}

nodes = rewards_full.keys()


def get_rank(r, nodes):
    v = np.array(list(r.values()))
    return [np.sum([r[node] <= v]) for node in nodes]


ranks_full = get_rank(mean_reward, nodes)
ranks_active_5 = get_rank(mean_reward_active_5, nodes)
ranks_random_5 = get_rank(mean_reward_random_5, nodes)

print("Active:")
print(mean_reward_active_5)
print("Random:")
print(mean_reward_random_5)

ax.plot([1, 15], [1, 15], color="black", zorder=0)
ax.scatter(
    ranks_full, ranks_active_5, label="Active inference"
)
ax.scatter(
    ranks_full, ranks_random_5, label="Even sampling", s=4
)
# ax.scatter(
#     [mean_reward[node] for node in mean_reward.keys()], [mean_reward_active_5[node] for node in mean_reward.keys()], label="Active inference"
# )
# ax.scatter(
#     [mean_reward[node] for node in mean_reward.keys()], [mean_reward_random_5[node] for node in mean_reward.keys()], label="Even sampling", s=4
# )
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
    from scipy.stats import beta as beta_dist

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
        3,
        5,
        figsize=(6.4 * 1.125, 4.8 * 1.125),
        sharex=True,
        sharey=True,
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

            y_z0 = beta_dist.pdf(x, alpha_z0, beta_z0)
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

            y_z1 = beta_dist.pdf(x, alpha_z1, beta_z1)
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
        question = "\n\\scriptsize ".join(
            textwrap.wrap(questions[node_id - 1], 22)
        )
        ax.text(
            0.0333,
            0.975,
            f"\\scriptsize ``{question}''",
            transform=ax.transAxes,
            ha="left",
            va="top",
            linespacing=1,
        )
        ax.set_ylim(0, 50)
        ax.set_xlim(0, 1)

        if i % 5 == 0:
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

        if i // 5 == 2:
            ax.tick_params(labelbottom=True)
            ax.set_xticks([0.25, 0.5, 0.75])
        else:
            ax.tick_params(labelbottom=False)

        if i // 5 == 1 and i % 5 == 0:
            ax.set_ylabel(
                "Correct answer probability\n$P(y_{j}=1|z)$"
            )

    fig.legend(
        loc="upper right",
        fontsize=8,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(1, 1.05),
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.25)

    plt.savefig(
        "output/posteriors.pdf", bbox_inches="tight"
    )

    return fig, axes


plot_y_distributions_by_z(active_5, mean_reward_active_5)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

df = pd.read_csv("output/utility.csv")
df.columns = ["efe", "eig", "r", "p"]

ax.fill_between(
    np.arange(len(df)),
    np.zeros(len(df)),
    df["r"] / df["efe"],
    alpha=0.5,
    label="$E[r_{\hat{j}}]/G_{\hat{j}}$\n(exploitation)",
)
ax.fill_between(
    np.arange(len(df)),
    df["r"] / df["efe"],
    np.ones(len(df)),
    alpha=0.5,
    label="$EIG(\hat{j})/G_{\hat{j}}$\n(exploration)",
)

ax.set_xlabel("Step $t$")
ax.set_ylabel("Contributions to $G_{\hat{j}}$")

fig.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.125),
    ncol=2,
    frameon=False,
)

fig.savefig("output/efe.pdf", bbox_inches="tight")
