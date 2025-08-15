import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib

from scipy.stats import beta as beta_dist
from scipy.special import expit, logit

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import resample

matplotlib.use("pgf")
plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
    },
)
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
import numpy as np
import seaborn as sns


def load_df(source):
    df = pd.read_csv(source)
    df = df[df["trial_maker_id"] == "optimal_test"]
    df["p"] = (
        df["p"].map(lambda s: json.loads(s)["1"] if not pd.isna(s) else None)
        if "p" in df.columns
        else None
    )
    return df


# Load the data
adaptive = load_df("output/KnowledgeTrial_adaptive.csv")
oracle = load_df("output/KnowledgeTrial_oracle.csv")
static = load_df("output/KnowledgeTrial_static.csv")

print(adaptive.groupby("participant_id")["node_id"].first().value_counts())


def plot_predictive_check(df):
    df = df.head(int(0.5*len(df)))
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
            n_failures = (1-bin_data["y"]).sum()

            # Calculate observed frequency
            observed_freq = (1+n_successes)/(2+n_successes+n_failures)
            observed_freqs.append(observed_freq)

            if n_in_bin > 0:
                conf_int = beta_dist.ppf([0.05/2, 1-0.05/2], 1+n_successes, 1+n_failures)
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
            ls="none"
        )

    ax.plot([0, 1], [0, 1], color="black")

    ax.set_xlabel("$q(y=1)$\n(Model posterior prediction)")
    ax.set_ylabel("$y$\n(Actual answer)")

    fig.savefig("output/evaluation_test.pdf", bbox_inches="tight")


plot_predictive_check(adaptive)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

n = adaptive.groupby("participant_id")["id"].count()
samples_adaptive = np.load("output/irt_samples_adaptive.npz")
samples_oracle = np.load("output/irt_samples_oracle.npz")
samples_static = np.load("output/irt_samples_static.npz")
theta_adaptive = samples_adaptive["theta"].mean(axis=0)
theta_oracle = samples_oracle["theta"].mean(axis=0)
theta_static = samples_static["theta"].mean(axis=0)

difficulty_oracle = samples_oracle["d"].mean(axis=0)

ax.axline((0, 0), slope=1, color="black", lw=1, alpha=0.5)
scatter = ax.scatter(
    theta_oracle,
    theta_adaptive,
    c=n,
    cmap=plt.cm.Blues,
    s=8,
    facecolors="none",
    edgecolors="black",
    lw=0.1,
    alpha=0.5,
)
ax.set_xlabel(r"Posterior $\bar{\theta_i}$ (oracle)")
ax.set_ylabel(r"Posterior $\bar{\theta_i}$ (adaptive)")

r2 = np.corrcoef(theta_oracle, theta_adaptive)[0, 1] ** 2
ax.text(
    0.075,
    0.95,
    f"$R^2={r2:.2f}$",
    transform=ax.transAxes,
    ha="left",
    va="top",
)
ax.text(
    0.0875,
    0.85,
    f"$\\bar{{n_i}}={n.mean():.1f}$",
    transform=ax.transAxes,
    ha="left",
    va="top",
)

# Add colorbar with legend
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(
    r"$n_{i}$\textsuperscript{trials}",
    rotation=270,
    labelpad=15,
)  # 'n' as the colorbar label
plt.show()
plt.savefig("output/theta_comparison.pdf", bbox_inches="tight")
plt.savefig("output/theta_comparison.png", bbox_inches="tight", dpi=300)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))
n = adaptive.groupby("node_id")["participant_id"].count()
n_oracle = oracle.groupby("node_id")["participant_id"].count()
ax.scatter(difficulty_oracle, n, label="Adaptive")
ax.scatter(difficulty_oracle, n_oracle, label="Oracle")
ax2 = ax.twinx()
sns.kdeplot(
    x=theta_oracle,
    label=r"$\theta_i$ distribution (oracle)",
    ax=ax2,
    hue_norm=(0, 200),
    bw_adjust=0.8,
    color="#4daf4a",
)
ax.set_xlabel("Challenge difficulty")
ax.set_ylabel("Number of trials")
ax.set_ylim(0, None)
fig.legend(
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.15),
    frameon=False,
)
plt.savefig("output/node_frequency.pdf", bbox_inches="tight")


fig, ax = plt.subplots(figsize=(3.2, 2.13333))
n = adaptive.groupby("participant_id")["id"].count()
ax.scatter(n.index, n, label="Adaptive", alpha=0.1)

# Prepare data for polynomial regression
X = n.index.values.reshape(-1, 1)
y = logit(n / (n.max() + 0.1))

# Create polynomial features (degree 2)
poly_features = PolynomialFeatures(degree=4, include_bias=True)
X_poly = poly_features.fit_transform(X)

# Fit linear regression with polynomial features
poly_model = LinearRegression()
poly_model.fit(X_poly, y)

# Generate smooth curve for plotting
X_smooth = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
X_smooth_poly = poly_features.transform(X_smooth)
predictions = []
for _ in range(2000):
    # Resample the data
    X_boot, y_boot = resample(X_poly, y, random_state=None)

    # Fit model on bootstrap sample
    model = LinearRegression()
    model.fit(X_boot, y_boot)

    # Predict on smooth grid
    y_pred = model.predict(X_smooth_poly)
    predictions.append(y_pred)

predictions = n.max() * expit(np.array(predictions))

# Calculate confidence intervals
mean_pred = np.mean(predictions, axis=0)
std_pred = np.std(predictions, axis=0)
ci_lower = np.percentile(predictions, 2.5, axis=0)
ci_upper = np.percentile(predictions, 97.5, axis=0)

# Plot the regression line and uncertainty
ax.plot(
    X_smooth.flatten(),
    mean_pred,
    linewidth=2,
    color="#377eb8",
    label="Polynomial fit",
)
ax.fill_between(
    X_smooth.flatten(),
    ci_lower,
    ci_upper,
    alpha=0.2,
    color="#377eb8",
    label="95% CI",
)
# Plot the regression line
# ax.plot(X_smooth.flatten(), expit(y_pred), linewidth=1.5)

ax.set_xlabel("Participant \#")
ax.set_ylabel("Trials per participant")
plt.savefig("output/trials_per_participant.pdf", bbox_inches="tight")
plt.savefig("output/trials_per_participant.png", bbox_inches="tight", dpi=300)

fig, ax = plt.subplots(figsize=(3.2, 2.13333))
n = adaptive.groupby("participant_id")["id"].count()
sd_adaptive = samples_adaptive["theta"].std(axis=0)
sd_oracle = samples_oracle["theta"].std(axis=0)
sd_static = samples_static["theta"].std(axis=0)
sd = samples_oracle["sigma_theta"]


def sd_to_entropy(sd):
    return 0.5 * np.log(2 * np.pi * sd * sd) + 0.5


entropy_adaptive = sd_to_entropy(sd_adaptive)
entropy_oracle = sd_to_entropy(sd_oracle)
entropy_static = sd_to_entropy(sd_static)
prior_entropy = sd_to_entropy(sd).mean(axis=0)

# Original violin plots
ax.violinplot(
    [entropy_oracle, entropy_adaptive, entropy_static],
    positions=[1, 2, 3],
)
# Prior entropy (position 0)
ax.plot([-0.125, 0.125], [prior_entropy] * 2, color="black")
ax.plot(
    [+0.125, ax.get_xlim()[1] - (-0.125 - ax.get_xlim()[0])],
    [prior_entropy] * 2,
    color="black",
    lw=0.5,
    ls="dashed",
)
ax.scatter([0], [prior_entropy], color="black", s=5)

# Add mean value text for the prior
ax.text(
    0,
    prior_entropy + 0.02,
    f"{prior_entropy:.2f}",
    ha="center",
    va="bottom",
)

# oracle posterior (position 2)
mean_val_oracle = entropy_oracle.mean()
ax.scatter([1], [mean_val_oracle], color="black", s=5)
ax.plot(
    [-0.125 + 1, 0.125 + 1],
    [mean_val_oracle] * 2,
    color="black",
)

# Add mean value text for oracle
ax.text(
    1,
    mean_val_oracle + 0.02,
    f"{mean_val_oracle:.2f}",
    ha="center",
    va="bottom",
)

# Adaptive posterior (position 2)
mean_val_adaptive = entropy_adaptive.mean()
ax.scatter([2], [mean_val_adaptive], color="black", s=5)
ax.plot(
    [-0.125 + 2, 0.125 + 2],
    [mean_val_adaptive] * 2,
    color="black",
)

# Add mean value text for adaptive
ax.text(
    2,
    mean_val_adaptive + 0.02,
    f"{mean_val_adaptive:.2f}",
    ha="center",
    va="bottom",
)

# Static posterior (position 3)
mean_val_static = entropy_static.mean()
ax.scatter([3], [mean_val_static], color="black", s=5)
ax.plot(
    [-0.125 + 3, 0.125 + 3],
    [mean_val_static] * 2,
    color="black",
)

# Add mean value text for adaptive
ax.text(
    3,
    mean_val_static + 0.02,
    f"{mean_val_static:.2f}",
    ha="center",
    va="bottom",
)

# Add arrows showing information gain

# Arrow from prior to oracle
oracle_info_gain = prior_entropy - mean_val_oracle
ax.annotate(
    "",
    xy=(0.75, mean_val_oracle),
    xytext=(0.75, prior_entropy),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)

# Information gain label for oracle
ax.text(
    0.65,
    (prior_entropy + mean_val_oracle) / 2,
    f"IG = {oracle_info_gain:.2f}",
    ha="center",
    va="center",
    fontsize=8,
    rotation=90,
)

# Arrow from prior to adaptive
adaptive_info_gain = prior_entropy - mean_val_adaptive
ax.annotate(
    "",
    xy=(1.75, mean_val_adaptive),
    xytext=(1.75, prior_entropy),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)

# Information gain label for adaptive
ax.text(
    1.65,
    (prior_entropy + mean_val_adaptive) / 2,
    f"IG = {adaptive_info_gain:.2f}",
    ha="center",
    va="center",
    fontsize=8,
    rotation=90,
)

# Arrow from prior to static
static_info_gain = prior_entropy - mean_val_static
ax.annotate(
    "",
    xy=(2.75, mean_val_static),
    xytext=(2.75, prior_entropy),
    arrowprops=dict(arrowstyle="->", lw=0.5),
)

# Information gain label for adaptive
ax.text(
    2.65,
    (prior_entropy + mean_val_static) / 2,
    f"IG = {static_info_gain:.2f}",
    ha="center",
    va="center",
    fontsize=8,
    rotation=90,
)

ax.set_xticks(
    [0, 1, 2, 3], ["No data\n(prior)", "Oracle", "Adaptive", "Static"]
)
ax.set_ylabel(r"$H(\theta_i)$ distribution")
ax.set_ylim(None, 2.0)

plt.savefig("output/entropy.pdf", bbox_inches="tight")
