"""Replay Experiment 1 on the human oracle and compare to the paper."""

import json
import logging
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import experiment as exp

logging.getLogger().setLevel(logging.WARNING)


class Participant:
    def __init__(self, participant_id):
        self.id = int(participant_id)
        self.var = type("V", (), {"z": None})()


def gauss_entropy(sd):
    sd = np.asarray(sd, dtype=float)
    return 0.5 * np.log(2.0 * np.pi * math.e * sd * sd)


def load_oracle():
    oracle = pd.read_csv("output/KnowledgeTrial_oracle.csv")
    oracle = oracle[oracle.trial_maker_id == "optimal_test"].copy()
    oracle["y_bin"] = oracle["y"].astype(str).str.lower().isin(
        ["true", "1", "1.0"]
    )
    answers = {
        (int(row.participant_id), int(row.item_id)): float(row.y_bin)
        for row in oracle.itertuples()
    }
    participants = sorted(int(p) for p in oracle.participant_id.unique())
    return answers, participants


def simulate(answers, participants, items):
    opt = exp.AdaptiveTesting(
        num_steps=400,
        num_samples=100,
        final_num_samples=10000,
        epsilon=0.04,
    )
    data = {
        "participants": {},
        "items": {item: {} for item in items},
        "y": {},
    }
    records = []
    n_trials = []
    trial_id = 0
    t0 = time.time()
    for k, pid in enumerate(participants, start=1):
        data["participants"][pid] = {"z": None}
        remaining = list(items)
        count = 0
        participant = Participant(pid)
        while remaining:
            node, _ = opt.get_optimal_node(remaining, participant, data)
            if node is None:
                break
            trial_id += 1
            count += 1
            y = answers[(pid, int(node))]
            data["y"][trial_id] = {
                "value": y,
                "participant_id": pid,
                "item_id": int(node),
            }
            records.append(
                {
                    "id": trial_id,
                    "participant_id": pid,
                    "item_id": int(node),
                    "y": bool(y),
                    "trial_maker_id": "optimal_test",
                }
            )
            remaining.remove(node)
        n_trials.append(count)
        print(
            f"p={pid:3d}  n={count:2d}  mean={np.mean(n_trials):.2f}  "
            f"last10={np.mean(n_trials[-10:]):.2f}  "
            f"t={time.time() - t0:.0f}s  obs={len(data['y'])}",
            flush=True,
        )
    return data, pd.DataFrame(records), np.asarray(n_trials)


def data_from_frame(df, participants, items):
    data = {
        "participants": {pid: {"z": None} for pid in participants},
        "items": {item: {} for item in items},
        "y": {},
    }
    for i, row in enumerate(df.itertuples(), start=1):
        data["y"][i] = {
            "value": float(row.y_bin if hasattr(row, "y_bin") else row.y),
            "participant_id": int(row.participant_id),
            "item_id": int(row.item_id),
        }
    return data


def fit_thetas(data):
    opt = exp.AdaptiveTesting(
        num_steps=400,
        num_samples=100,
        final_num_samples=1000,
        epsilon=0.04,
    )
    opt.update_posterior(data)
    order = list(data["participants"])
    means = {
        pid: float(opt.theta_means[i]) for i, pid in enumerate(order)
    }
    sds = {pid: float(opt.theta_sds[i]) for i, pid in enumerate(order)}
    return means, sds


def plot_trials(n, paper_n, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    idx = np.arange(1, len(n) + 1)
    ax.scatter(idx, n, s=12, alpha=0.4, label="Journal replay")
    ax.scatter(
        np.arange(1, len(paper_n) + 1),
        paper_n,
        s=12,
        alpha=0.25,
        label="Paper simulation (repo)",
    )
    window = 15
    kernel = np.ones(window) / window
    if len(n) >= window:
        smooth = np.convolve(n, kernel, mode="valid")
        ax.plot(
            np.arange(window, len(n) + 1),
            smooth,
            color="C0",
            lw=2,
            label=f"Replay rolling mean ({window})",
        )
    if len(paper_n) >= window:
        smooth_p = np.convolve(paper_n, kernel, mode="valid")
        ax.plot(
            np.arange(window, len(paper_n) + 1),
            smooth_p,
            color="C1",
            lw=2,
            ls="--",
            label="Paper rolling mean",
        )
    ax.axhline(9.6, color="black", lw=0.8, ls=":", label="Paper mean 9.6")
    ax.set_xlabel("Participant #")
    ax.set_ylabel("Trials per participant")
    ax.set_ylim(0, 16)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_theta(theta_oracle, theta_adapt, n, path):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    pids = sorted(theta_oracle)
    x = np.array([theta_oracle[p] for p in pids])
    y = np.array([theta_adapt[p] for p in pids])
    n_by_pid = {pid: int(n[i]) for i, pid in enumerate(sorted(theta_oracle))}
    counts = np.array([n_by_pid[p] for p in pids])
    ax.axline((0, 0), slope=1, color="black", lw=1, alpha=0.4)
    scatter = ax.scatter(x, y, c=counts, cmap="Blues", s=16, alpha=0.8)
    r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
    ax.text(0.05, 0.95, f"$R^2={r2:.2f}$", transform=ax.transAxes, va="top")
    ax.text(
        0.05,
        0.87,
        f"$\\bar{{n}}={counts.mean():.1f}$",
        transform=ax.transAxes,
        va="top",
    )
    ax.set_xlabel(r"Posterior $\bar\theta_i$ (oracle)")
    ax.set_ylabel(r"Posterior $\bar\theta_i$ (adaptive)")
    fig.colorbar(scatter, ax=ax, label="trials")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return r2


def main():
    answers, participants = load_oracle()
    items = list(range(15))
    print(f"oracle participants={len(participants)} items={len(items)}")

    paper = pd.read_csv("output/KnowledgeTrial_adaptive.csv")
    paper = paper[paper.trial_maker_id == "optimal_test"]
    paper_n = (
        paper.groupby("participant_id").size().reindex(participants).to_numpy()
    )
    print(
        f"paper adaptive mean={paper_n.mean():.2f} "
        f"first20={paper_n[:20].mean():.2f} last20={paper_n[-20:].mean():.2f}"
    )

    data, trials, n = simulate(answers, participants, items)
    os.makedirs("data", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    trials.to_csv("data/journal_sim_trials.csv", index=False)
    pd.Series(n, index=participants, name="n_trials").to_csv(
        "data/journal_sim_n_trials.csv"
    )

    print("=== TRIAL COUNTS ===")
    print(
        f"replay mean={n.mean():.2f} sd={n.std():.2f} "
        f"min={n.min()} max={n.max()}"
    )
    print(f"first20={n[:20].mean():.2f} last20={n[-20:].mean():.2f}")
    print(f"reduction vs 15: {100 * (1 - n.mean() / 15):.1f}%")
    print("paper simulation mean=9.61, reduction ~36%")

    oracle_rows = []
    for (pid, item), y in answers.items():
        oracle_rows.append(
            {"participant_id": pid, "item_id": item, "y": y, "y_bin": y}
        )
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_data = data_from_frame(oracle_df, participants, items)

    print("Fitting variational posteriors on adaptive and oracle data...")
    theta_adapt, sd_adapt = fit_thetas(data)
    theta_oracle, sd_oracle = fit_thetas(oracle_data)

    H_prior = gauss_entropy(2.0)
    ig_adapt = H_prior - gauss_entropy(list(sd_adapt.values())).mean()
    ig_oracle = H_prior - gauss_entropy(list(sd_oracle.values())).mean()
    print("=== INFORMATION GAIN (variational 1PL) ===")
    print(f"H_prior={H_prior:.3f}")
    print(
        f"IG adaptive={ig_adapt:.3f}  IG oracle={ig_oracle:.3f}  "
        f"ratio={ig_adapt / ig_oracle:.3f}"
    )
    print("paper HMC: IG 0.76 vs 0.79 (ratio 0.96)")

    r2 = plot_theta(
        theta_oracle,
        theta_adapt,
        n,
        "output/theta_comparison_journal.png",
    )
    plot_trials(n, paper_n, "output/trials_per_participant_journal.png")
    print(f"R^2 adaptive vs oracle theta = {r2:.3f} (paper 0.96)")
    print("wrote output/trials_per_participant_journal.png")
    print("wrote output/theta_comparison_journal.png")


if __name__ == "__main__":
    main()
