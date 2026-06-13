#!/usr/bin/env python3
import os
from argparse import ArgumentParser
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path

# Avoid nested BLAS/OpenMP thread pools inside parallel run
# workers. Callers can override these environment variables.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import matplotlib
import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.contrib.oed.eig import marginal_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.distributions.constraints import positive
from tqdm import tqdm


matplotlib.use("pdf")
DEFAULT_OUTPUT_DIR = Path("output/eig_vs_fisher_pl2")
DEFAULT_MAX_RUN_WORKERS = 4


def output_dir(args):
    return args.output_dir / f"{args.items}_{args.trials}"


def plate_stack(name, sizes):
    stack = ExitStack()
    for index, size in enumerate(reversed(sizes), start=1):
        stack.enter_context(
            pyro.plate(
                f"{name}_{index}",
                size,
                dim=-index,
            )
        )
    return stack


def oracle_lookup(oracle):
    return {
        (row["participant_id"], row["item_id"]): row
        for row in oracle.to_dict(orient="records")
    }


@dataclass
class PL2Population:
    theta: torch.Tensor
    difficulty: torch.Tensor
    sensitivity: torch.Tensor
    intercept: torch.Tensor


def sample_truncated_normal(
    mean,
    std,
    size,
    generator,
    lower=0.0,
):
    samples = torch.empty(size)
    remaining = torch.ones(size, dtype=torch.bool)
    while remaining.any():
        draws = torch.normal(
            mean=mean,
            std=std,
            size=(int(remaining.sum()),),
            generator=generator,
        )
        accepted = draws > lower
        if accepted.any():
            indices = remaining.nonzero(
                as_tuple=False
            ).flatten()
            samples[indices[accepted]] = draws[accepted]
            remaining[indices[accepted]] = False

    return samples


def draw_population(num_participants, num_items, seed):
    generator = torch.Generator().manual_seed(seed)

    return PL2Population(
        theta=torch.normal(
            mean=0.0,
            std=1.0,
            size=(num_participants,),
            generator=generator,
        ),
        difficulty=4
        * (
            torch.rand(
                size=(num_items,),
                generator=generator,
            )
            - 0.5
        ),
        sensitivity=sample_truncated_normal(
            mean=1.0,
            std=1.0,
            size=(num_items,),
            generator=generator,
            lower=0.0,
        ),
        intercept=torch.normal(
            mean=0.0,
            std=0.5,
            size=(),
            generator=generator,
        ),
    )


def generate_oracle(population, run_id, seed):
    rng = np.random.default_rng(seed)
    records = []

    for participant_id in range(len(population.theta)):
        for item_id in range(len(population.difficulty)):
            true_logit = (
                population.sensitivity[item_id]
                * (
                    population.theta[participant_id]
                    - population.difficulty[item_id]
                )
                + population.intercept
            )
            true_prob = float(torch.sigmoid(true_logit))
            y = int(rng.binomial(1, true_prob))

            records.append(
                {
                    "scenario": "oracle",
                    "run_id": run_id,
                    "participant_id": participant_id,
                    "item_id": item_id,
                    "y": y,
                    "true_prob": true_prob,
                    "true_theta": float(
                        population.theta[participant_id]
                    ),
                    "true_difficulty": float(
                        population.difficulty[item_id]
                    ),
                    "true_sensitivity": float(
                        population.sensitivity[item_id]
                    ),
                    "true_intercept": float(
                        population.intercept
                    ),
                }
            )

    return pd.DataFrame.from_records(records)


class AdaptivePL2Learner:
    def __init__(
        self,
        num_participants,
        num_items,
        posterior_steps,
        eig_steps,
        eig_samples,
        eig_final_samples,
        start_lr,
        end_lr,
        seed,
    ):
        self.num_participants = num_participants
        self.num_items = num_items
        self.posterior_steps = posterior_steps
        self.eig_steps = eig_steps
        self.eig_samples = eig_samples
        self.eig_final_samples = eig_final_samples
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.seed = seed

        self.prior_mean_theta = torch.tensor(0.0)
        self.prior_sd_theta = torch.tensor(2.0)
        self.prior_mean_difficulty = torch.tensor(0.0)
        self.prior_sd_difficulty = torch.tensor(1.0)
        self.prior_mean_log_sensitivity = torch.tensor(0.0)
        self.prior_sd_log_sensitivity = torch.tensor(1.0)
        self.prior_mean_intercept = torch.tensor(0.0)
        self.prior_sd_intercept = torch.tensor(1.0)

        pyro.clear_param_store()
        pyro.set_rng_seed(seed)
        self._reset_posterior()

    def _reset_posterior(self):
        self.theta_means = torch.full(
            (self.num_participants,),
            self.prior_mean_theta,
        )
        self.theta_sds = torch.full(
            (self.num_participants,),
            self.prior_sd_theta,
        )
        self.difficulty_means = torch.full(
            (self.num_items,),
            self.prior_mean_difficulty,
        )
        self.difficulty_sds = torch.full(
            (self.num_items,),
            self.prior_sd_difficulty,
        )
        self.log_sensitivity_means = torch.full(
            (self.num_items,),
            self.prior_mean_log_sensitivity,
        )
        self.log_sensitivity_sds = torch.full(
            (self.num_items,),
            self.prior_sd_log_sensitivity,
        )
        self.intercept_mean = (
            self.prior_mean_intercept.clone()
        )
        self.intercept_sd = self.prior_sd_intercept.clone()

    def _model(self, participants, items):
        thetas = pyro.sample(
            "thetas",
            dist.Normal(
                self.prior_mean_theta,
                self.prior_sd_theta,
            )
            .expand((self.num_participants,))
            .to_event(1),
        )
        difficulties = pyro.sample(
            "difficulties",
            dist.Normal(
                self.prior_mean_difficulty,
                self.prior_sd_difficulty,
            )
            .expand((self.num_items,))
            .to_event(1),
        )
        log_sensitivities = pyro.sample(
            "log_sensitivities",
            dist.Normal(
                self.prior_mean_log_sensitivity,
                self.prior_sd_log_sensitivity,
            )
            .expand((self.num_items,))
            .to_event(1),
        )
        intercept = pyro.sample(
            "intercept",
            dist.Normal(
                self.prior_mean_intercept,
                self.prior_sd_intercept,
            ),
        )

        sensitivities = torch.exp(log_sensitivities)
        item_ids = items.long()
        logits = (
            sensitivities[item_ids]
            * (
                thetas[participants.long()]
                - difficulties[item_ids]
            )
            + intercept
        )
        return pyro.sample(
            "y",
            dist.Bernoulli(logits=logits).to_event(1),
        )

    def _guide(self, participants, items):
        theta_means = pyro.param(
            "theta_means",
            torch.full(
                (self.num_participants,),
                self.prior_mean_theta,
            ),
        )
        theta_sds = pyro.param(
            "theta_sds",
            torch.full(
                (self.num_participants,),
                self.prior_sd_theta,
            ),
            constraint=positive,
        )
        pyro.sample(
            "thetas",
            dist.Normal(theta_means, theta_sds).to_event(1),
        )

        difficulty_means = pyro.param(
            "difficulty_means",
            torch.full(
                (self.num_items,),
                self.prior_mean_difficulty,
            ),
        )
        difficulty_sds = pyro.param(
            "difficulty_sds",
            torch.full(
                (self.num_items,),
                self.prior_sd_difficulty,
            ),
            constraint=positive,
        )
        pyro.sample(
            "difficulties",
            dist.Normal(
                difficulty_means,
                difficulty_sds,
            ).to_event(1),
        )

        log_sensitivity_means = pyro.param(
            "log_sensitivity_means",
            torch.full(
                (self.num_items,),
                self.prior_mean_log_sensitivity,
            ),
        )
        log_sensitivity_sds = pyro.param(
            "log_sensitivity_sds",
            torch.full(
                (self.num_items,),
                self.prior_sd_log_sensitivity,
            ),
            constraint=positive,
        )
        pyro.sample(
            "log_sensitivities",
            dist.Normal(
                log_sensitivity_means,
                log_sensitivity_sds,
            ).to_event(1),
        )

        intercept_mean = pyro.param(
            "intercept_mean",
            self.prior_mean_intercept.clone(),
        )
        intercept_sd = pyro.param(
            "intercept_sd",
            self.prior_sd_intercept.clone(),
            constraint=positive,
        )
        pyro.sample(
            "intercept",
            dist.Normal(intercept_mean, intercept_sd),
        )

    def _make_design_model(self, participant_id):
        def model(design):
            with plate_stack("plate", design.shape[:-1]):
                theta = pyro.sample(
                    "theta",
                    dist.Normal(
                        self.theta_means[participant_id],
                        self.theta_sds[participant_id],
                    ),
                )
                theta = theta.unsqueeze(-1)

                item_idx = design.squeeze(-1).long()
                difficulties = pyro.sample(
                    "difficulties",
                    dist.Normal(
                        self.difficulty_means[item_idx],
                        self.difficulty_sds[item_idx],
                    ),
                ).unsqueeze(-1)
                log_sensitivities = pyro.sample(
                    "log_sensitivities",
                    dist.Normal(
                        self.log_sensitivity_means[
                            item_idx
                        ],
                        self.log_sensitivity_sds[item_idx],
                    ),
                ).unsqueeze(-1)

                intercept = pyro.sample(
                    "intercept",
                    dist.Normal(
                        self.intercept_mean,
                        self.intercept_sd,
                    ),
                ).unsqueeze(-1)

                sensitivities = torch.exp(log_sensitivities)
                logits = (
                    sensitivities * (theta - difficulties)
                    + intercept
                )
                return pyro.sample(
                    "y",
                    dist.Bernoulli(logits=logits).to_event(
                        1
                    ),
                )

        return model

    def _marginal_guide(
        self,
        design,
        observation_labels,
        target_labels,
    ):
        q_logit = pyro.param(
            "q_logit",
            torch.zeros(design.shape[-2:]),
        )
        pyro.sample(
            "y",
            dist.Bernoulli(logits=q_logit).to_event(1),
        )

    def update_posterior(self, records):
        if len(records) == 0:
            self._reset_posterior()
            return

        participants = torch.tensor(
            [
                record["participant_id"]
                for record in records
            ],
            dtype=torch.long,
        )
        items = torch.tensor(
            [record["item_id"] for record in records],
            dtype=torch.long,
        )
        responses = torch.tensor(
            [record["y"] for record in records],
            dtype=torch.float,
        )

        pyro.clear_param_store()
        conditioned_model = pyro.condition(
            self._model,
            {"y": responses},
        )
        svi = SVI(
            conditioned_model,
            self._guide,
            Adam({"lr": 0.02}),
            loss=Trace_ELBO(),
        )

        for _ in range(self.posterior_steps):
            svi.step(participants, items)

        self.theta_means = (
            pyro.param("theta_means").detach().clone()
        )
        self.theta_sds = (
            pyro.param("theta_sds").detach().clone()
        )
        self.difficulty_means = (
            pyro.param("difficulty_means").detach().clone()
        )
        self.difficulty_sds = (
            pyro.param("difficulty_sds").detach().clone()
        )
        self.log_sensitivity_means = (
            pyro.param("log_sensitivity_means")
            .detach()
            .clone()
        )
        self.log_sensitivity_sds = (
            pyro.param("log_sensitivity_sds")
            .detach()
            .clone()
        )
        self.intercept_mean = (
            pyro.param("intercept_mean").detach().clone()
        )
        self.intercept_sd = (
            pyro.param("intercept_sd").detach().clone()
        )

    def select_item(self, participant_id, candidates):
        pyro.clear_param_store()
        design_model = self._make_design_model(
            participant_id
        )
        candidate_designs = torch.tensor(
            candidates,
            dtype=torch.float,
        ).unsqueeze(-1)

        optimizer = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": self.start_lr},
                "gamma": (self.end_lr / self.start_lr)
                ** (1 / self.eig_steps),
            }
        )
        eig = marginal_eig(
            design_model,
            candidate_designs,
            "y",
            [
                "theta",
                "difficulties",
                "log_sensitivities",
                "intercept",
            ],
            num_samples=self.eig_samples,
            num_steps=self.eig_steps,
            guide=self._marginal_guide,
            optim=optimizer,
            final_num_samples=self.eig_final_samples,
        )

        predictive_probs = torch.special.expit(
            pyro.param("q_logit")
        ).detach()
        predictive_probs = predictive_probs.squeeze(-1)

        best_idx = torch.argmax(eig)
        best_item = candidates[int(best_idx)]
        best_eig = float(eig[best_idx].detach())
        best_predictive_prob = float(
            predictive_probs[best_idx]
        )

        return best_item, best_eig, best_predictive_prob


class PointwiseFisherPL2Learner(AdaptivePL2Learner):
    def select_item(self, participant_id, candidates):
        candidate_ids = torch.tensor(
            candidates, dtype=torch.long
        )
        sensitivities = torch.exp(
            self.log_sensitivity_means[candidate_ids]
        )
        logits = (
            sensitivities
            * (
                self.theta_means[participant_id]
                - self.difficulty_means[candidate_ids]
            )
            + self.intercept_mean
        )
        predictive_probs = torch.special.expit(logits)
        fisher = (
            sensitivities**2
            * predictive_probs
            * (1 - predictive_probs)
        )

        best_idx = torch.argmax(fisher)
        best_item = candidates[int(best_idx)]
        best_fisher = float(fisher[best_idx].detach())
        best_predictive_prob = float(
            predictive_probs[best_idx].detach()
        )

        return best_item, best_fisher, best_predictive_prob


def make_learner(learner_cls, population, args, seed):
    return learner_cls(
        num_participants=len(population.theta),
        num_items=len(population.difficulty),
        posterior_steps=args.posterior_steps,
        eig_steps=args.eig_steps,
        eig_samples=args.eig_samples,
        eig_final_samples=args.eig_final_samples,
        start_lr=args.start_lr,
        end_lr=args.end_lr,
        seed=seed,
    )


def simulate_strategy(
    learner,
    population,
    oracle_responses,
    run_id,
    trials,
    scenario,
):
    records = []
    participant_ids = tqdm(
        range(len(population.theta)),
        desc=f"{scenario} run {run_id}",
        unit="participant",
    )

    for participant_id in participant_ids:
        unanswered_items = set(
            range(len(population.difficulty))
        )

        for trial_index in range(trials):
            candidates = sorted(unanswered_items)
            item_id, score, predictive_prob = (
                learner.select_item(
                    participant_id,
                    candidates,
                )
            )
            unanswered_items.remove(item_id)

            oracle_row = oracle_responses[
                (participant_id, item_id)
            ]
            y = int(oracle_row["y"])

            record = {
                "scenario": scenario,
                "run_id": run_id,
                "participant_id": participant_id,
                "trial_index": trial_index,
                "item_id": item_id,
                "y": y,
                "true_prob": oracle_row["true_prob"],
                "predictive_prob": predictive_prob,
                "eig": np.nan,
                "pointwise_fisher": np.nan,
                "true_theta": oracle_row["true_theta"],
                "true_difficulty": oracle_row[
                    "true_difficulty"
                ],
                "true_sensitivity": oracle_row[
                    "true_sensitivity"
                ],
                "true_intercept": oracle_row[
                    "true_intercept"
                ],
            }
            if scenario == "eig":
                record["eig"] = score
            elif scenario == "pointwise_fisher":
                record["pointwise_fisher"] = score

            records.append(record)
            learner.update_posterior(records)

    return pd.DataFrame.from_records(records)


def simulate_run(run_id, args):
    population = draw_population(
        num_participants=args.participants,
        num_items=args.items,
        seed=args.seed + run_id,
    )
    oracle = generate_oracle(
        population=population,
        run_id=run_id,
        seed=args.seed + 500 * (run_id + 1),
    )
    responses = oracle_lookup(oracle)

    eig_learner = make_learner(
        AdaptivePL2Learner,
        population,
        args,
        seed=args.seed + 1_000 * (run_id + 1),
    )
    fisher_learner = make_learner(
        PointwiseFisherPL2Learner,
        population,
        args,
        seed=args.seed + 2_000 * (run_id + 1),
    )

    eig = simulate_strategy(
        learner=eig_learner,
        population=population,
        oracle_responses=responses,
        run_id=run_id,
        trials=args.trials,
        scenario="eig",
    )
    fisher = simulate_strategy(
        learner=fisher_learner,
        population=population,
        oracle_responses=responses,
        run_id=run_id,
        trials=args.trials,
        scenario="pointwise_fisher",
    )

    return oracle, eig, fisher


def write_run_outputs(run_id, args, run_output_dir):
    torch.set_num_threads(args.torch_threads)
    print(f"Starting run {run_id}", flush=True)

    oracle, eig, fisher = simulate_run(run_id, args)
    outputs = dict(
        zip(
            ["oracle", "eig", "pointwise_fisher"],
            [oracle, eig, fisher],
        )
    )
    written = []
    for scenario, filename in output_specs(run_id):
        df = outputs[scenario]
        path = run_output_dir / filename
        df.to_csv(path, index=False)
        written.append((scenario, len(df), path))

    return run_id, written


def run_simulations(run_ids, args, run_output_dir):
    if not run_ids:
        return

    max_workers = args.run_workers
    if max_workers is None:
        max_workers = min(
            len(run_ids),
            os.cpu_count() or 1,
            DEFAULT_MAX_RUN_WORKERS,
        )

    if max_workers == 1:
        for run_id in run_ids:
            _, written = write_run_outputs(
                run_id, args, run_output_dir
            )
            for _, row_count, path in written:
                print(f"Wrote {row_count} rows to {path}")
        return

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context("spawn"),
    ) as executor:
        futures = {
            executor.submit(
                write_run_outputs,
                run_id,
                args,
                run_output_dir,
            ): run_id
            for run_id in run_ids
        }
        for future in as_completed(futures):
            run_id = futures[future]
            _, written = future.result()
            print(f"Completed run {run_id}")
            for _, row_count, path in written:
                print(f"Wrote {row_count} rows to {path}")


def output_specs(run_id):
    return [
        (
            "oracle",
            f"responses_oracle_run_{run_id:03d}.csv",
        ),
        ("eig", f"responses_eig_run_{run_id:03d}.csv"),
        (
            "pointwise_fisher",
            f"responses_pointwise_fisher_run_{run_id:03d}.csv",
        ),
    ]


def complete_run_exists(run_output_dir, run_id):
    return all(
        (run_output_dir / filename).exists()
        for _, filename in output_specs(run_id)
    )


def next_run_ids(run_output_dir, n_runs):
    run_ids = []
    candidate = 0
    while len(run_ids) < n_runs:
        if complete_run_exists(run_output_dir, candidate):
            print(f"Skipping existing run {candidate}")
        else:
            run_ids.append(candidate)
        candidate += 1

    return run_ids


def run_id_from_path(path):
    return int(path.stem.rsplit("_", maxsplit=1)[-1])


def remove_prefix(value, prefix):
    if value.startswith(prefix):
        return value[len(prefix) :]

    return value


def scenario_from_path(path):
    return remove_prefix(path.stem, "responses_").rsplit(
        "_run_",
        maxsplit=1,
    )[0]


def hmc_estimates_path(run_output_dir):
    return run_output_dir / "hmc_parameter_estimates.csv"


def response_paths(run_output_dir):
    return sorted(
        list(
            run_output_dir.glob(
                "responses_oracle_run_*.csv"
            )
        )
        + list(
            run_output_dir.glob("responses_eig_run_*.csv")
        )
        + list(
            run_output_dir.glob(
                "responses_pointwise_fisher_run_*.csv"
            )
        )
    )


def parse_condition_dir(path):
    parts = path.name.split("_")
    if len(parts) != 2:
        return None

    try:
        return {
            "items": int(parts[0]),
            "trials": int(parts[1]),
        }
    except ValueError:
        return None


def condition_dirs(args):
    dirs = []
    if args.output_dir.exists():
        for path in sorted(args.output_dir.iterdir()):
            if not path.is_dir():
                continue
            condition = parse_condition_dir(path)
            if condition is None or not response_paths(
                path
            ):
                continue
            dirs.append((path, condition))

    # Keep plot mode useful when the caller points directly at one
    # simulation folder rather than the experiment root.
    direct_condition = parse_condition_dir(args.output_dir)
    if not dirs and direct_condition is not None:
        if response_paths(args.output_dir):
            dirs.append((args.output_dir, direct_condition))

    return dirs


def fit_pl2_hmc(
    df,
    draws,
    tune,
    chains,
    cores,
    target_accept,
    seed,
):
    try:
        import pymc as pm  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Plot mode requires PyMC. Install `pymc` in the "
            "environment used to run this script."
        ) from exc

    unique_participants = sorted(
        df["participant_id"].unique()
    )
    unique_items = sorted(df["item_id"].unique())
    participant_index = {
        participant_id: index
        for index, participant_id in enumerate(
            unique_participants
        )
    }
    item_index = {
        item_id: index
        for index, item_id in enumerate(unique_items)
    }
    participant_ids = (
        df["participant_id"]
        .map(participant_index)
        .to_numpy()
    )
    item_ids = df["item_id"].map(item_index).to_numpy()
    responses = df["y"].astype(int).to_numpy()

    coords = {
        "participant": unique_participants,
        "item": unique_items,
    }
    with pm.Model(coords=coords):
        theta = pm.Normal(
            "theta",
            mu=0,
            sigma=2,
            dims="participant",
        )
        difficulty = pm.Normal(
            "difficulty",
            mu=0,
            sigma=1,
            dims="item",
        )
        log_sensitivity = pm.Normal(
            "log_sensitivity",
            mu=0,
            sigma=1,
            dims="item",
        )
        sensitivity = pm.Deterministic(
            "sensitivity",
            pm.math.exp(log_sensitivity),
            dims="item",
        )
        intercept = pm.Normal("intercept", mu=0, sigma=1)
        logits = (
            sensitivity[item_ids]
            * (
                theta[participant_ids]
                - difficulty[item_ids]
            )
            + intercept
        )
        pm.Bernoulli(
            "y", logit_p=logits, observed=responses
        )

        inference_data = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=True,
        )

    theta_mean = (
        inference_data.posterior["theta"]
        .mean(dim=("chain", "draw"))
        .to_numpy()
    )
    difficulty_mean = (
        inference_data.posterior["difficulty"]
        .mean(dim=("chain", "draw"))
        .to_numpy()
    )
    sensitivity_mean = (
        inference_data.posterior["sensitivity"]
        .mean(dim=("chain", "draw"))
        .to_numpy()
    )

    theta_estimates = pd.DataFrame(
        {
            "parameter": "ability",
            "parameter_id": unique_participants,
            "estimate": theta_mean,
        }
    )
    difficulty_estimates = pd.DataFrame(
        {
            "parameter": "difficulty",
            "parameter_id": unique_items,
            "estimate": difficulty_mean,
        }
    )
    sensitivity_estimates = pd.DataFrame(
        {
            "parameter": "sensitivity",
            "parameter_id": unique_items,
            "estimate": sensitivity_mean,
        }
    )
    return pd.concat(
        [
            theta_estimates,
            difficulty_estimates,
            sensitivity_estimates,
        ],
        ignore_index=True,
    )


def compute_hmc_estimates(run_output_dir, args):
    cache_path = hmc_estimates_path(run_output_dir)
    if cache_path.exists() and not args.force_hmc:
        return pd.read_csv(cache_path)

    rows = []
    for source_index, path in enumerate(
        response_paths(run_output_dir)
    ):
        run_id = run_id_from_path(path)
        scenario = scenario_from_path(path)
        df = pd.read_csv(path)
        print(
            f"Fitting {scenario} run {run_id} from {path}"
        )
        estimates = fit_pl2_hmc(
            df=df,
            draws=args.hmc_draws,
            tune=args.hmc_tune,
            chains=args.hmc_chains,
            cores=args.hmc_cores,
            target_accept=args.hmc_target_accept,
            seed=args.seed
            + 10_000 * (run_id + 1)
            + source_index,
        )
        estimates["scenario"] = scenario
        estimates["run_id"] = run_id
        rows.append(estimates)

    if not rows:
        raise FileNotFoundError(
            f"No response CSV files found in {run_output_dir}."
        )

    estimates = pd.concat(rows, ignore_index=True)
    estimates.to_csv(cache_path, index=False)
    return estimates


def hmc_squared_errors(estimates):
    oracle = estimates[
        estimates["scenario"] == "oracle"
    ].rename(columns={"estimate": "oracle_estimate"})
    scenario_estimates = estimates[
        estimates["scenario"].isin(
            ["eig", "pointwise_fisher"]
        )
    ].rename(columns={"estimate": "scenario_estimate"})

    merged = scenario_estimates.merge(
        oracle[
            [
                "run_id",
                "parameter",
                "parameter_id",
                "oracle_estimate",
            ]
        ],
        on=["run_id", "parameter", "parameter_id"],
        how="inner",
    )

    merged["error"] = (
        merged["scenario_estimate"]
        - merged["oracle_estimate"]
    )
    merged["squared_error"] = merged["error"] ** 2

    return merged[
        [
            "run_id",
            "scenario",
            "parameter",
            "parameter_id",
            "scenario_estimate",
            "oracle_estimate",
            "error",
            "squared_error",
        ]
    ]


def rmse_summary(errors):
    summary = (
        errors.groupby(
            ["items", "trials", "scenario", "parameter"],
            as_index=False,
        )
        .agg(
            mse=("squared_error", "mean"),
            sem_mse=("squared_error", "sem"),
            n_estimates=("squared_error", "size"),
        )
        .fillna({"sem_mse": 0.0})
    )
    summary["mean_rmse"] = np.sqrt(summary["mse"])
    summary["sem_rmse"] = np.where(
        summary["mean_rmse"] > 0,
        summary["sem_mse"] / (2 * summary["mean_rmse"]),
        0.0,
    )

    return summary


def plot_hmc_rmse(args):
    import matplotlib.pyplot as plt

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for run_output_dir, condition in condition_dirs(args):
        estimates = compute_hmc_estimates(
            run_output_dir, args
        )
        errors = hmc_squared_errors(estimates)
        errors["items"] = condition["items"]
        errors["trials"] = condition["trials"]
        rows.append(errors)

        condition_errors_path = (
            run_output_dir / "hmc_squared_errors.csv"
        )
        errors.to_csv(condition_errors_path, index=False)
        print(
            f"Wrote {len(errors)} rows to {condition_errors_path}"
        )

    if not rows:
        raise FileNotFoundError(
            f"No simulation outputs found in {args.output_dir}."
        )

    errors = pd.concat(rows, ignore_index=True)
    errors_path = (
        args.output_dir / "hmc_squared_errors_all.csv"
    )
    errors.to_csv(errors_path, index=False)
    print(f"Wrote {len(errors)} rows to {errors_path}")

    summary = rmse_summary(errors)
    summary_path = args.output_dir / "hmc_rmse_all.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {len(summary)} rows to {summary_path}")

    labels = {
        "eig": "EIG",
        "pointwise_fisher": "Fisher information",
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.6))
    multiple_trial_counts = summary["trials"].nunique() > 1
    plot_specs = [
        ("ability", "Ability RMSE"),
        ("difficulty", "Difficulty RMSE"),
        ("sensitivity", "Sensitivity RMSE"),
    ]
    for ax, (parameter, title) in zip(axes, plot_specs):
        scenarios = ["eig", "pointwise_fisher"]
        parameter_scores = summary[
            summary["parameter"] == parameter
        ]
        for scenario in scenarios:
            scenario_scores = parameter_scores[
                parameter_scores["scenario"] == scenario
            ]
            for (
                trials,
                trial_scores,
            ) in scenario_scores.groupby("trials"):
                trial_scores = trial_scores.sort_values(
                    "items"
                )
                label = labels[scenario]
                if multiple_trial_counts:
                    label = f"{label}, n={trials}"
                ax.errorbar(
                    trial_scores["items"],
                    trial_scores["mean_rmse"],
                    yerr=trial_scores["sem_rmse"],
                    marker="o",
                    capsize=3,
                    label=label,
                )

        ax.set_title(title)
        ax.set_xlabel("Number of items")
        ax.set_ylabel("RMSE vs oracle HMC")
        ax.set_xticks(
            sorted(parameter_scores["items"].unique())
        )

    axes[0].legend(frameon=False)

    fig.tight_layout()
    pdf_path = args.output_dir / "hmc_rmse_by_items.pdf"
    png_path = args.output_dir / "hmc_rmse_by_items.png"
    fig.savefig(pdf_path)
    print(f"Wrote {pdf_path}")
    fig.savefig(png_path, dpi=300)
    print(f"Wrote {png_path}")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Compare EIG item selection with pointwise Fisher "
            "selection on the same PL2 oracle."
        )
    )
    parser.add_argument(
        "--participants", type=int, default=200
    )
    parser.add_argument("--items", type=int, default=15)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument(
        "--run-workers",
        type=int,
        default=None,
        help=(
            "Number of runs to execute in parallel. Defaults to "
            "min(n-runs, CPU count, 4). Use 1 for sequential "
            "execution."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help=(
            "Torch intra-op threads per run worker. Keep this low "
            "when running many runs in parallel."
        ),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--posterior-steps", type=int, default=400
    )
    parser.add_argument(
        "--eig-steps", type=int, default=400
    )
    parser.add_argument(
        "--eig-samples", type=int, default=100
    )
    parser.add_argument(
        "--eig-final-samples",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--start-lr", type=float, default=0.1
    )
    parser.add_argument(
        "--end-lr", type=float, default=0.001
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Fit HMC on existing oracle, EIG, and Fisher outputs "
            "and plot RMSE against the oracle HMC estimates."
        ),
    )
    parser.add_argument(
        "--hmc-draws", type=int, default=1000
    )
    parser.add_argument(
        "--hmc-tune", type=int, default=1000
    )
    parser.add_argument("--hmc-chains", type=int, default=4)
    parser.add_argument("--hmc-cores", type=int, default=4)
    parser.add_argument(
        "--hmc-target-accept",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--force-hmc",
        action="store_true",
        help="Refit HMC even when cached estimates exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1")
    if args.trials < 1:
        raise ValueError("--trials must be at least 1")
    if args.trials > args.items:
        raise ValueError("--trials must be <= --items")
    if (
        args.run_workers is not None
        and args.run_workers < 1
    ):
        raise ValueError("--run-workers must be at least 1")
    if args.torch_threads < 1:
        raise ValueError(
            "--torch-threads must be at least 1"
        )

    if args.plot:
        plot_hmc_rmse(args)
        return

    run_output_dir = output_dir(args)
    run_output_dir.mkdir(parents=True, exist_ok=True)

    run_ids = next_run_ids(run_output_dir, args.n_runs)
    run_simulations(run_ids, args, run_output_dir)


if __name__ == "__main__":
    main()
