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
    return df


# Load the data
adaptive = load_df('output/KnowledgeTrial_adaptive.csv')
oracle = load_df('output/KnowledgeTrial_oracle.csv')

adaptive = adaptive[adaptive["trial_maker_id"] == "optimal_test"]
oracle = oracle[oracle["trial_maker_id"] == "optimal_test"]

print(adaptive.groupby("participant_id")["node_id"].first().value_counts())

fig, ax = plt.subplots(figsize=(3.2, 2.13333))

n = adaptive.groupby("participant_id")["id"].count()
samples_adaptive = np.load("output/irt_samples_adaptive.npz")
samples_oracle = np.load("output/irt_samples_oracle.npz")
theta_adaptive = samples_adaptive['theta'].mean(axis=0)
theta_oracle = samples_oracle['theta'].mean(axis=0)

difficulty_oracle = samples_oracle["d"].mean(axis=0)

ax.axline((0, 0), slope=1, color="black", lw=1, alpha=0.5)
scatter = ax.scatter(
    theta_oracle, theta_adaptive, c=n, cmap=plt.cm.Blues, s=8,
    facecolors='none', edgecolors='black', lw=0.1, alpha=0.5,
)
ax.set_xlabel(r"Posterior $\theta_i$ (oracle)")
ax.set_ylabel(r"Posterior $\theta_i$ (adaptive)")

r2 = np.corrcoef(theta_oracle, theta_adaptive)[0, 1] ** 2
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
    r'$n_{i}$\textsuperscript{trials}', rotation=270, labelpad=15,
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
    x=theta_oracle, label=r"$\theta_i$ distribution (oracle)", ax=ax2,
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

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.special import expit, logit
from sklearn.utils import resample

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
    X_smooth.flatten(), mean_pred, linewidth=2, color='#377eb8',
    label='Polynomial fit',
)
ax.fill_between(
    X_smooth.flatten(), ci_lower, ci_upper, alpha=0.2, color='#377eb8',
    label='95% CI',
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
sd = samples_oracle["sigma_theta"]


def sd_to_entropy(sd):
    return 0.5 * np.log(2 * np.pi * sd * sd) + 0.5


entropy_adaptive = sd_to_entropy(sd_adaptive)
entropy_oracle = sd_to_entropy(sd_oracle)
prior_entropy = sd_to_entropy(sd).mean(axis=0)

# Original violin plots
ax.violinplot(
    [entropy_adaptive, entropy_oracle], positions=[1, 2],
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

# oracle posterior (position 2)
mean_val_oracle = entropy_oracle.mean()
ax.scatter([2], [mean_val_oracle], color="black", s=5)
ax.plot(
    [-0.125 + 2, 0.125 + 2], [mean_val_oracle] * 2,
    color="black",
)

# Add mean value text for oracle
ax.text(
    2, mean_val_oracle + 0.02, f'{mean_val_oracle:.2f}', ha='center',
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

# Arrow from prior to oracle
oracle_info_gain = prior_entropy - mean_val_oracle
ax.annotate(
    '', xy=(1.75, mean_val_oracle), xytext=(1.75, prior_entropy),
    arrowprops=dict(arrowstyle='->', lw=0.5),
)

# Information gain label for oracle
ax.text(
    1.65, (prior_entropy + mean_val_oracle) / 2,
    f'IG = {oracle_info_gain:.2f}',
    ha='center', va='center', fontsize=8,
    rotation=90,
)

ax.set_xticks([0, 1, 2], ["No data (prior)", "Adaptive", "Oracle"])
ax.set_ylabel(r"$H(\theta_i)$ distribution")
ax.set_ylim(None, 2.0)

plt.savefig("output/entropy.pdf", bbox_inches="tight")
