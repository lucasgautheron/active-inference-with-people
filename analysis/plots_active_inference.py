import pandas as pd
import numpy as np
import json

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

from scipy.stats import beta as beta_dist

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample


def load_df(source, shift_node_id=False, samples=None):
    df = pd.read_csv(source)
    df = df[df["trial_maker_id"] == "optimal_treatment"]
    df["p"] = (
        df["p"].map(lambda s: json.loads(s)["1"] if not pd.isna(s) else None)
        if "p" in df.columns
        else None
    )
    df = df[df["finalized"] == True]

    if "deployment" in source:
        participants = pd.read_csv("output/Participant_deployment.csv")
        participants = participants[participants["progress"] == 1]
        df = df[df["participant_id"].isin(participants["id"])]

    if shift_node_id:
        df["node_id"] = df["node_id"] - 15
        df = df[df["node_id"] >= 1]

    df["z"] = df["z"].astype(int)
    # df["z"] = df["z"].map(lambda s: json.loads(s)["value"])

    if samples is not None:
        rows = []
        counter = {node: 0 for node in df["node_id"].unique()}
        for participant_id in df["participant_id"].unique():
            nodes = sorted(counter.keys(), key=lambda node: counter[node])[
                :samples
            ]
            rows.append(
                df[
                    (df["participant_id"] == participant_id)
                    & (df["node_id"].isin(nodes))
                ]
            )
            for node in nodes:
                counter[node] += 1

        df = pd.concat(rows)
        print(df)

    print(df)

    return df


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
    z = np.random.binomial(np.ones(n_samples, dtype=int), Phi)

    nodes_ids = sorted(df["node_id"].unique())

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
        p_y = alpha / (alpha + beta) * y + beta / (alpha + beta) * (1 - y)

        p_y_given_phi = p_y_given_phi[z, np.arange(n_samples)]
        p_y = p_y[z, np.arange(n_samples)]
        EIG = np.mean(np.log(p_y_given_phi / p_y))

        p_z = z.mean()

        gamma = 0.2
        p_z = 0.5
        U = gamma * (
            p_z * phi[1]
            + (1 - p_z) * (1 - phi[0])
            - p_z * (1 - phi[1])
            - (1 - p_z) * phi[0]
        )

        rewards[node_id] = EIG + U
        eig[node_id] = EIG
        utility[node_id] = U

    return utility


# Load the data
adaptive = load_df("output/KnowledgeTrial_adaptive_treatment.csv")
deployment = load_df("output/KnowledgeTrial_deployment.csv")


def plot_predictive_check(df, output):
    df = df.head(int(0.5 * len(df)))
    fig, ax = plt.subplots(figsize=(3.2, 2.333))

    ax.scatter(df["p"], df["y"], alpha=0.075, s=5, edgecolors="black", lw=0.2)

    n_bins = 8
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = (bin_lowers + bin_uppers) / 2

    observed_freqs = []
    ci_lowers = []
    ci_uppers = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Get data points in this bin
        in_bin = (df["p"] >= bin_lower) & (df["p"] < bin_upper)
        if bin_upper == 1.0:  # Include the upper boundary for the last bin
            in_bin = (df["p"] >= bin_lower) & (df["p"] <= bin_upper)

        bin_data = df[in_bin]
        n_in_bin = len(bin_data)
        bin_counts.append(n_in_bin)

        if n_in_bin > 0:
            n_successes = bin_data["y"].sum()
            n_failures = (1 - bin_data["y"]).sum()

            # Calculate observed frequency
            observed_freq = (1 + n_successes) / (2 + n_successes + n_failures)
            observed_freqs.append(observed_freq)

            if n_in_bin > 0:
                conf_int = beta_dist.ppf(
                    [0.05 / 2, 1 - 0.05 / 2], 1 + n_successes, 1 + n_failures
                )
                ci_lowers.append(conf_int[0])
                ci_uppers.append(conf_int[1])
            else:
                ci_lowers.append(observed_freq)
                ci_uppers.append(observed_freq)
        else:
            observed_freqs.append(np.nan)
            ci_lowers.append(np.nan)
            ci_uppers.append(np.nan)

    observed_freqs = np.array(observed_freqs)
    ci_lowers = np.array(ci_lowers)
    ci_uppers = np.array(ci_uppers)
    bin_counts = np.array(bin_counts)

    has_data = ~np.isnan(observed_freqs) & (bin_counts > 0)

    if np.any(has_data):
        ax.errorbar(
            bin_centers[has_data],
            observed_freqs[has_data],
            yerr=[
                observed_freqs[has_data] - ci_lowers[has_data],
                ci_uppers[has_data] - observed_freqs[has_data],
            ],
            fmt="o",
            capsize=3,
            capthick=1,
            markersize=4,
            ls="none",
        )

    ax.plot([0, 1], [0, 1], color="black")

    ax.set_xlabel("$p(y=1)$\n(Model posterior prediction)")
    ax.set_ylabel("$y$\n(Actual answer)")

    fig.savefig(output, bbox_inches="tight")


plot_predictive_check(adaptive, "output/evaluation_treatment.pdf")
plot_predictive_check(deployment, "output/evaluation_treatment_deployment.pdf")

oracle = load_df("output/KnowledgeTrial_oracle_treatment.csv", True)
static = load_df("output/KnowledgeTrial_oracle_treatment.csv", True, samples=5)

rewards_oracle = expected_free_energy(oracle)
rewards_adaptive = expected_free_energy(adaptive)
rewards_deployment = expected_free_energy(deployment)
rewards_static = expected_free_energy(static)

mean_reward = {
    node: np.mean(rewards_oracle[node]) for node in rewards_oracle.keys()
}
low_reward = {
    node: np.quantile(rewards_oracle[node], q=0.1 / 2)
    for node in rewards_oracle.keys()
}
high_reward = {
    node: np.quantile(rewards_oracle[node], q=1 - 0.1 / 2)
    for node in rewards_oracle.keys()
}
order = sorted(
    list(mean_reward.keys()),
    key=lambda node: mean_reward[node],
)


mean_reward_adaptive = {
    node: np.mean(rewards_adaptive[node]) for node in rewards_adaptive.keys()
}
mean_reward_deployment = {
    node: np.mean(rewards_deployment[node])
    for node in rewards_deployment.keys()
}
mean_reward_static = {
    node: np.mean(rewards_static[node]) for node in rewards_static.keys()
}
low_reward_adaptive = {
    node: np.quantile(rewards_adaptive[node], q=0.05 / 2)
    for node in rewards_adaptive.keys()
}
low_reward_static = {
    node: np.quantile(rewards_static[node], q=0.05 / 2)
    for node in rewards_static.keys()
}

high_reward_adaptive = {
    node: np.quantile(rewards_adaptive[node], q=1 - 0.05 / 2)
    for node in rewards_adaptive.keys()
}
high_reward_static = {
    node: np.quantile(rewards_static[node], q=1 - 0.05 / 2)
    for node in rewards_static.keys()
}


fig, ax = plt.subplots(figsize=(3.2, 2.13333))

nodes = list(mean_reward.keys())


def cumulative_frequency(df, mean_reward, output):
    cmap = plt.get_cmap("RdBu")

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
    amplitude = np.max(np.abs(list(mean_reward.values())))
    norm = plt.Normalize(
        vmin=-amplitude,
        vmax=+amplitude,
    )
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # Required for ScalarMappable

    for node in unique_nodes:
        color = cmap(norm(mean_reward[node]))
        ax.plot(cumulative_data[node], color=color)

    ax.set_xlabel("Step $t$")
    ax.set_ylabel("Cumulative frequency\nof treatment $j$\nat step $t$")
    cbar = plt.colorbar(sm, ax=ax)  # Use the ScalarMappable here
    cbar.set_label(
        r"$E[r_j]_{oracle}$",
        rotation=90,
        labelpad=15,
    )
    fig.savefig(
        f"{output}.pdf",
        bbox_inches="tight",
    )
    fig.savefig(f"{output}.png", bbox_inches="tight", dpi=300)


cumulative_frequency(adaptive, mean_reward, "output/cumulative_node_frequency")
cumulative_frequency(
    deployment,
    mean_reward_deployment,
    "output/cumulative_node_frequency_deployment",
)
cumulative_frequency(
    static, mean_reward, "output/cumulative_node_frequency_random"
)


def frequency(adaptive, static, mean_reward, output):
    cmap = plt.get_cmap("RdBu")

    unique_nodes = sorted(static["node_id"].unique())
    freq_adaptive = adaptive["node_id"].value_counts().to_dict()
    frequencies_adaptive = [freq_adaptive[node] for node in unique_nodes]
    freq_static = static["node_id"].value_counts().to_dict()
    frequencies_static = [freq_static[node] for node in unique_nodes]
    rewards = [mean_reward[node] for node in unique_nodes]

    fig, ax = plt.subplots(figsize=(3.2, 2.13333))
    ax.scatter(rewards, frequencies_adaptive, label="Adaptive")
    ax.scatter(rewards, frequencies_static, label="Even sampling")
    fig.legend(
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        frameon=False,
    )
    fig.savefig(f"{output}.pdf", bbox_inches="tight")


frequency(adaptive, static, mean_reward, "output/frequency")

nodes = rewards_oracle.keys()


def get_rank(r, nodes):
    v = np.array(list(r.values()))
    return [np.sum([r[node] <= v]) for node in nodes]


ranks_oracle = get_rank(mean_reward, nodes)
ranks_adaptive = get_rank(mean_reward_adaptive, nodes)
ranks_static = get_rank(mean_reward_static, nodes)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

ax.plot([1, 15], [1, 15], color="black", zorder=0)
ax.scatter(ranks_oracle, ranks_adaptive, label="Active inference")
ax.scatter(ranks_oracle, ranks_static, label="Even sampling", s=4)
ax.set_xlabel("Treatment rank\nin the oracle setup")
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


fig, ax = plt.subplots(figsize=(3.2, 2.13333))

ax.scatter(
    get_rank(mean_reward, nodes),
    [mean_reward_adaptive[node] for node in mean_reward.keys()],
    label="Active inference",
    s=6,
)
ax.errorbar(
    get_rank(mean_reward, nodes),
    [mean_reward_adaptive[node] for node in mean_reward.keys()],
    yerr=(
        [
            mean_reward_adaptive[node] - low_reward_adaptive[node]
            for node in mean_reward.keys()
        ],
        [
            high_reward_adaptive[node] - mean_reward_adaptive[node]
            for node in mean_reward.keys()
        ],
    ),
    ls="none",
    capsize=3,
    capthick=1,
)
ax.scatter(
    get_rank(mean_reward, nodes),
    [mean_reward_static[node] for node in mean_reward.keys()],
    label="Even sampling",
    s=6,
)
bars = ax.errorbar(
    get_rank(mean_reward, nodes),
    [mean_reward_static[node] for node in mean_reward.keys()],
    yerr=(
        [
            mean_reward_static[node] - low_reward_static[node]
            for node in mean_reward.keys()
        ],
        [
            high_reward_static[node] - mean_reward_static[node]
            for node in mean_reward.keys()
        ],
    ),
    ls="none",
    capsize=3,
    capthick=0.5,
    elinewidth=0.5,
)

ax.set_xticks(
    [1, 5, 10, 15],
    ["1st\n(best)", "5th", "10th", "15th\n(worst)"],
)
ax.set_xlabel("Treatment rank (oracle)")
ax.set_ylabel("Estimated treatment utility\n$E[r_j]$")

fig.legend(
    frameon=False,
    bbox_to_anchor=(0.5, 1.05),
    loc="upper center",
    ncol=2,
)
plt.savefig("output/reward.pdf", bbox_inches="tight")


def plot_y_distributions_by_z(df, mean_reward, output):
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

    questions = df_clean.drop_duplicates("node_id").set_index("node_id")
    questions = (
        questions["definition"]
        .apply(lambda s: json.loads(s)["question"])
        .to_dict()
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
            successes_z0 = z0_data["y"].sum()  # Count of True values
            failures_z0 = len(z0_data) - successes_z0  # Count of False values
            alpha_z0 = successes_z0 + 1  # Beta prior with alpha=1, beta=1
            beta_z0 = failures_z0 + 1

            y_z0 = beta_dist.pdf(x, alpha_z0, beta_z0)
            ax.plot(
                x,
                y_z0,
                color="#377eb8",
                linewidth=2,
                label=("High-scool or less ($z=0$)" if i == 0 else None),
            )
            ax.fill_between(x, y_z0, alpha=0.3, color="#377eb8")

        # Plot beta distribution for z=1
        if len(z1_data) > 0:
            successes_z1 = z1_data["y"].sum()  # Count of True values
            failures_z1 = len(z1_data) - successes_z1  # Count of False values
            alpha_z1 = successes_z1 + 1  # Beta prior with alpha=1, beta=1
            beta_z1 = failures_z1 + 1

            y_z1 = beta_dist.pdf(x, alpha_z1, beta_z1)
            ax.plot(
                x,
                y_z1,
                color="#ff7f00",
                linewidth=2,
                label=("College ($z=1$)" if i == 0 else None),
            )
            ax.fill_between(x, y_z1, alpha=0.3, color="#ff7f00")

        # Customize subplot
        ax.set_title(
            f"$E[r_j]={mean_reward[node_id]:.2f}$",
        )
        question = "\n\\scriptsize ".join(textwrap.wrap(questions[node_id], 22))
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
            ax.set_ylabel("Correct answer probability\n$P(y_{j}=1|z)$")

    fig.legend(
        loc="upper right",
        fontsize=8,
        ncol=2,
        frameon=False,
        bbox_to_anchor=(1, 1.05),
    )

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.25)

    plt.savefig(output, bbox_inches="tight")

    return fig, axes


plot_y_distributions_by_z(oracle, mean_reward, "output/posteriors_oracle.pdf")
plot_y_distributions_by_z(
    adaptive, mean_reward_adaptive, "output/posteriors.pdf"
)
plot_y_distributions_by_z(
    deployment, mean_reward_deployment, "output/posteriors_deployment.pdf"
)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

df = pd.read_csv("output/utility_adaptive.csv")
df.columns = ["efe", "eig", "r"]

ax.fill_between(
    np.arange(len(df)),
    np.zeros(len(df)),
    df["r"],
    alpha=0.5,
    label="$E[r_{\hat{j}}]$\n(exploitation)",
)
ax.fill_between(
    np.arange(len(df)),
    np.maximum(df["r"], 0),
    np.maximum(df["r"], 0) + df["eig"],
    alpha=0.5,
    label="$EIG(\hat{j})$\n(exploration)",
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
fig.savefig("output/efe.png", bbox_inches="tight", dpi=300)
