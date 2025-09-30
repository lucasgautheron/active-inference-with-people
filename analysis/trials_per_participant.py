import json
import matplotlib

import matplotlib.pyplot as plt
from scipy.special import expit, logit
from cmdstanpy import CmdStanModel
import pandas as pd

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
import numpy as np
import seaborn as sns


def load_df(source, samples=None):
    df = pd.read_csv(source)
    df = df[df["trial_maker_id"] == "optimal_test"]
    df["p"] = (
        df["p"].map(
            lambda s: (
                json.loads(s)["1"]
                if not pd.isna(s)
                else None
            )
        )
        if "p" in df.columns
        else None
    )
    if "deployment" in source:
        participants = pd.read_csv(
            "output/Participant_deployment.csv"
        )
        participants = participants[
            participants["progress"] == 1
        ]
        df = df[
            df["participant_id"].isin(participants["id"])
        ]

    if samples is not None:
        df = df.sample(n=samples)

    return df


# Load the data
adaptive = load_df("output/KnowledgeTrial_adaptive.csv")
deployment = load_df("output/KnowledgeTrial_deployment.csv")
oracle = load_df("output/KnowledgeTrial_oracle_fast.csv")
static = load_df(
    "output/KnowledgeTrial_oracle_fast.csv", len(adaptive)
)


def trials_per_participant(df, output):
    """
    Plot trials per participant with polynomial mixture model using Stan.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns 'participant_id' and 'id'
    output : str
        Path to save the output figure
    """
    fig, ax = plt.subplots(figsize=(3.2, 2.13333))

    # Count trials per participant
    n = df.groupby("participant_id")["id"].count()
    ax.scatter(n.index, n, label="Adaptive", alpha=0.1)

    # Prepare data for Stan
    x = n.index.values
    y = n.values

    # Standardize x for numerical stability
    x_standardized = x / 200

    # Prepare Stan data
    stan_data = {
        "N": len(x),
        "y": y.astype(int),
        "x": x_standardized,
    }

    # Compile and fit the model
    model = CmdStanModel(
        stan_file="analysis/trials_per_participant.stan"
    )
    fit = model.sample(
        data=stan_data,
        chains=4,
        parallel_chains=4,
        iter_warmup=1000,
        iter_sampling=2000,
        show_console=False,
    )

    # Extract posterior samples for p component
    beta_samples = fit.stan_variable("beta")
    beta1_samples = fit.stan_variable("beta1")
    beta2_samples = fit.stan_variable("beta2")
    beta3_samples = fit.stan_variable("beta3")

    # Extract posterior samples for pi (mixture probability)
    gamma_samples = fit.stan_variable("gamma")
    gamma1_samples = fit.stan_variable("gamma1")
    gamma2_samples = fit.stan_variable("gamma2")

    # Generate smooth curve for plotting
    x_smooth = np.linspace(x.min(), x.max(), 100)
    x_smooth_standardized = x_smooth / 200

    # Calculate predictions for each posterior sample
    n_samples = len(beta_samples)
    predictions = np.zeros((n_samples, len(x_smooth)))
    pi_predictions = np.zeros((n_samples, len(x_smooth)))

    for i in range(n_samples):
        # Calculate mu (mean probability for binomial)
        linear_predictor_mu = (
            beta_samples[i]
            + beta1_samples[i] * x_smooth_standardized
            + beta2_samples[i] * x_smooth_standardized**2
            + beta3_samples[i] * x_smooth_standardized**3
        )
        mu = expit(linear_predictor_mu)

        # Calculate pi (mixture probability)
        linear_predictor_pi = (
            + gamma_samples[i]
            + gamma1_samples[i] * x_smooth_standardized
            + gamma2_samples[i] * x_smooth_standardized**2
        )
        pi = expit(linear_predictor_pi)
        pi_predictions[i, :] = pi

        # Expected value: pi * E[Binomial(15, p)] + (1-pi) * 15
        # where E[Binomial(15, p)] ≈ 15 * mu for our model
        predictions[i, :] = pi * 15 * mu + (1 - pi) * 15

    # Calculate posterior summaries for expected trials
    mean_pred = np.mean(predictions, axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)

    # Plot the regression line and uncertainty for expected trials
    ax.plot(
        x_smooth,
        mean_pred,
        linewidth=2,
        color="#377eb8",
        label="Mixture model fit",
    )
    ax.fill_between(
        x_smooth,
        ci_lower,
        ci_upper,
        alpha=0.2,
        color="#377eb8",
        label="95% CI",
    )

    ax.set_xlabel("Participant \#")
    ax.set_ylabel("Trials per participant")

    plt.tight_layout()
    plt.savefig(output, bbox_inches="tight")
    plt.close()

    return fit


trials_per_participant(
    adaptive, "output/trials_per_participant_bayes.pdf"
)
trials_per_participant(
    deployment,
    "output/trials_per_participant_bayes_deployment.pdf",
)
