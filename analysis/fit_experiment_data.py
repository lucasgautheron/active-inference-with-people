import pandas as pd
import numpy as np
import cmdstanpy
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# Load and prepare the data
def load_and_prepare_data(csv_file, samples=None):
    """Load the knowledge trial data and prepare it for IRT modeling"""

    df = pd.read_csv(csv_file)
    df = df[df["trial_maker_id"] == "optimal_test"]

    if samples is not None:
        df = df.sample(samples)

    # Convert y column to binary (handle string and boolean values)
    df["y_binary"] = df["y"].apply(
        lambda x: 1 if x == "True" or x == True else 0,
    )

    # Create sequential participant and item IDs for Stan (1-indexed)
    unique_participants = sorted(df["participant_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    # Create mapping dictionaries
    participant_map = {
        pid: idx + 1 for idx, pid in enumerate(unique_participants)
    }
    item_map = {iid: idx + 1 for idx, iid in enumerate(unique_items)}

    # Apply mappings
    df["stan_participant_id"] = df["participant_id"].map(participant_map)
    df["stan_item_id"] = df["item_id"].map(item_map)

    print(f"Data preparation complete:")
    print(f"  {len(unique_participants)} participants")
    print(f"  {len(unique_items)} items")
    print(f"  {len(df)} total responses")
    print(f"  Overall accuracy: {df['y_binary'].mean():.3f}")

    return df, participant_map, item_map, unique_participants, unique_items


# Create the Stan model code
def create_stan_model():
    """Create the IRT model in Stan (2PL model)"""

    stan_code = """
    data {
        int<lower=1> N;                    // number of observations
        int<lower=1> J;                    // number of participants
        int<lower=1> K;                    // number of items
        array[N] int<lower=1,upper=J> jj;  // participant for observation n
        array[N] int<lower=1,upper=K> kk;  // item for observation n
        array[N] int<lower=0,upper=1> y;   // yness for observation n
    }

    parameters {
        vector[J] theta;           // ability parameters (participants)
        vector[K] d;            // difficulty parameters (items)
        real<lower=0> sigma_theta;
    }

    model {
        // Priors
        theta ~ normal(0, sigma_theta); // standard normal abilities
        d ~ normal(0, 1); // difficulty parameters

        for (n in 1:N) {
            y[n] ~ bernoulli_logit(theta[jj[n]] - d[kk[n]]);
        }
        
        sigma_theta ~ exponential(1);
    }

    generated quantities {
        // Posterior predictive checks
        array[N] int y_rep;
        vector[N] log_lik;

        for (n in 1:N) {
            real p = inv_logit(theta[jj[n]] - d[kk[n]]);
            y_rep[n] = bernoulli_rng(p);
            log_lik[n] = bernoulli_logit_lpmf(y[n] | theta[jj[n]] - d[kk[n]]);
        }
    }
    """

    return stan_code


def fit_irt_model(
    df,
    chains=4,
    iter_warmup=1000,
    iter_sampling=1000,
    parallel_chains=4,
):
    """Fit the IRT model using cmdstanpy"""

    # Prepare data for Stan
    stan_data = {
        "N": len(df),
        "J": df["stan_participant_id"].nunique(),
        "K": df["stan_item_id"].nunique(),
        "jj": df["stan_participant_id"].tolist(),
        "kk": df["stan_item_id"].tolist(),
        "y": df["y_binary"].tolist(),
    }

    print("Stan data prepared:")
    for key, value in stan_data.items():
        if isinstance(value, list):
            print(f"  {key}: list of length {len(value)}")
        else:
            print(f"  {key}: {value}")

    # Write Stan model to file
    stan_code = create_stan_model()
    model_file = "output/irt_model.stan"
    with open(model_file, "w") as f:
        f.write(stan_code)

    print(f"\nStan model written to {model_file}")

    # Compile and fit the model
    print("Compiling Stan model...")
    model = cmdstanpy.CmdStanModel(stan_file=model_file)

    print("Fitting model...")
    fit = model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        seed=42,
        show_progress=True,
    )

    return fit, stan_data


def save_stan_samples(fit, output_file):
    """Save all Stan samples to npz file"""

    # Get all variable names from the fit
    stan_vars = fit.stan_variables()

    print(f"\nSaving Stan samples to {output_file}")
    print("Variables saved:")

    # Save all variables to npz file
    np.savez_compressed(output_file, **stan_vars)

    # Print info about saved variables
    for var_name, var_data in stan_vars.items():
        print(f"  {var_name}: shape {var_data.shape}")

    print(f"\nSamples successfully saved to {output_file}")
    return stan_vars


# Main execution function
def main():
    """Main function to run the complete IRT analysis"""

    print("Starting IRT Analysis with Stan")
    print("=" * 50)

    # 1. Load and prepare data
    df, participant_map, item_map, unique_participants, unique_items = (
        load_and_prepare_data(
            "output/KnowledgeTrial_adaptive_fast.csv",
        )
    )
    fit, stan_data = fit_irt_model(
        df,
        chains=4,
        iter_warmup=1000,
        iter_sampling=2000,
    )
    stan_vars = save_stan_samples(fit, "output/irt_samples_adaptive.npz")
    n_adaptive_responses = len(df)

    df, participant_map, item_map, unique_participants, unique_items = (
        load_and_prepare_data(
            "output/KnowledgeTrial_oracle_fast.csv",
        )
    )
    fit, stan_data = fit_irt_model(
        df,
        chains=4,
        iter_warmup=1000,
        iter_sampling=2000,
    )
    stan_vars = save_stan_samples(fit, "output/irt_samples_oracle.npz")

    df, participant_map, item_map, unique_participants, unique_items = (
        load_and_prepare_data(
            "output/KnowledgeTrial_oracle_fast.csv",
            samples=n_adaptive_responses
        )
    )
    fit, stan_data = fit_irt_model(
        df,
        chains=4,
        iter_warmup=1000,
        iter_sampling=2000,
    )
    stan_vars = save_stan_samples(fit, "output/irt_samples_static.npz")

# Run the analysis
if __name__ == "__main__":
    main()
