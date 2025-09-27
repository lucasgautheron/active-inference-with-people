import pandas as pd
import numpy as np
from scipy import stats

import matplotlib
from matplotlib import pyplot as plt

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

from matplotlib import pyplot as plt

response_times = pd.read_csv(
    "output/KnowledgeTrial_oracle_treatment.csv"
)


def get_response_time(participant_id, node_id):
    """
    Get the time it took for a participant to answer a specific node.

    Args:
        participant_id (int): The participant ID
        node_id (int): The node/item ID

    Returns:
        float or None: Response time if found, None otherwise
    """
    # Load the oracle trials data

    # Filter for the specific participant and node
    result = response_times[
        (response_times["participant_id"] == participant_id)
        & (response_times["node_id"] == node_id)
    ]

    if result.empty:
        raise Exception()
        return None

    # Return the time_taken value
    return result["time_taken"].iloc[0]


df = pd.read_csv(
    "output/duration.csv",
    names=[
        "participant_id",
        "iteration",
        "node_id",
        "eig",
        "U",
        "G",
        "p",
        "delta",
    ],
)

# Group by participant_id and iteration
grouped = df.groupby(["participant_id", "iteration"])

print(
    f"\nNumber of (participant_id, iteration) groups: {len(grouped)}"
)


# Function to find argmax differences for each group
def find_argmax_differences(group):
    # Find the index (node_id) with maximum eig value
    eig_argmax_idx = group["eig"].idxmax()
    eig_argmax_node = group.loc[eig_argmax_idx, "node_id"]
    eig_argmax_p = group.loc[eig_argmax_idx, "p"]

    # Find the index (node_id) with maximum G value
    g_argmax_idx = group["G"].idxmax()
    g_argmax_node = group.loc[g_argmax_idx, "node_id"]
    g_argmax_p = group.loc[g_argmax_idx, "p"]

    # Check if they differ
    differs = eig_argmax_node != g_argmax_node

    return pd.Series(
        {
            "eig_argmax_node": eig_argmax_node,
            "g_argmax_node": g_argmax_node,
            "differs": differs,
            "eig_max_value": group["eig"].max(),
            "g_max_value": group["G"].max(),
            "group_size": len(group),
            "eig_argmax_p": eig_argmax_p,
            "g_argmax_p": g_argmax_p,
        }
    )


# Apply the function to each group
results = grouped.apply(
    find_argmax_differences
).reset_index()

# Count total instances where argmax differs
total_differences = results["differs"].sum()
total_groups = len(results)

differences_only = results[results["differs"] == True]
print(differences_only)

df = df.drop_duplicates(subset=["node_id"], keep="last")
print(df)

fig, axes = plt.subplots(
    figsize=(3.2 * 1.75, 2.333),
    ncols=2,
    nrows=1,
    gridspec_kw={"width_ratios": [2, 1]},
)
ax = axes[0]
ax.scatter(df["delta"], df["p"])
ax.set_xlabel("Relative question difficulty")
ax.set_ylabel(
    "Probability of slow answer\n$q(\\tau_j>20s)$"
)

results["response_time_avoided"] = results.apply(
    lambda row: get_response_time(
        row["participant_id"], row["eig_argmax_node"]
    )
    - get_response_time(
        row["participant_id"], row["g_argmax_node"]
    ),
    axis=1,
)

# time_delta = results[results["differs"] == True][
#     "response_time_avoided"
# ]

time_delta = results["response_time_avoided"]

# Calculate Bayesian 95% credible interval using conjugate priors
n = len(time_delta)
sample_mean = np.mean(time_delta)
sample_var = np.var(time_delta, ddof=1)

# Prior parameters (weakly informative)
mu_0 = 0.0  # prior mean
kappa_0 = 1.0  # prior precision parameter
nu_0 = 1.0  # prior degrees of freedom
sigma2_0 = 10.0  # prior variance

# Posterior parameters
kappa_n = kappa_0 + n
nu_n = nu_0 + n
mu_n = (kappa_0 * mu_0 + n * sample_mean) / kappa_n
sigma2_n = (1 / nu_n) * (
    nu_0 * sigma2_0
    + (n - 1) * sample_var
    + (kappa_0 * n / kappa_n) * (sample_mean - mu_0) ** 2
)

# μ|data ~ t(μ_n, σ²_n/κ_n, ν_n)
posterior_scale = np.sqrt(sigma2_n / kappa_n)
t_critical = stats.t.ppf(0.975, df=nu_n)
ci_lower = mu_n - t_critical * posterior_scale
ci_upper = mu_n + t_critical * posterior_scale
mean_delta = mu_n

ax = axes[1]

# Plot the mean with error bars representing 95% CI
ax.errorbar(
    0,
    mean_delta,
    yerr=[[mean_delta - ci_lower], [ci_upper - mean_delta]],
    fmt="o",
    capsize=5,
    capthick=2,
)
ax.set_xlim(-0.5, 2)
ax.set_ylim(-0.5, 2)
ax.axhline(0, color="black", lw=1)
ax.text(
    0.95,
    0.95,
    f"\\scriptsize $\mu={mean_delta:.3f}$",
    ha="right",
    va="top",
    transform=ax.transAxes,
)
ax.text(
    0.95,
    0.85,
    f"\\scriptsize $CI_{{95\\%}}=[{ci_lower:.3f}, {ci_upper:.3f}]$",
    ha="right",
    va="top",
    transform=ax.transAxes,
)
ax.set_xticks([], [])
ax.set_ylabel("Average time saved per trial\n$\\delta t$ (s)")
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()

fig.savefig(
    "output/response_times_vs_difficulty.pdf",
    bbox_inches="tight",
)
