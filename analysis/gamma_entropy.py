"""
Compute rolling entropy of the arm selection distribution.

For each row (step t) of a simulation, compute the entropy of the
empirical distribution of selected arms over the last K steps:

    p_j = (# times arm j was selected in [t-K+1, t]) / K
    H(t) = -sum_j p_j * log(p_j)    (in nats, 0*log0 = 0)

This measures how uniformly the agent is exploring:
    H = log(K)  →  pure exploration (uniform)
    H = 0       →  pure exploitation (one arm always chosen)

Loads: active_inference_sims.csv
Saves: active_inference_entropy.csv
"""

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import time

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
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

INPUT_FILE  = "output/gamma_active_inference_sims.csv"
OUTPUT_FILE = "output/gamma_active_inference_entropy.csv"

# ── entropy from counts ───────────────────────────────────────────────────────
def entropy_from_counts(counts):
    """Shannon entropy (nats) from a count array. 0*log0 = 0."""
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.sum(np.where(p > 0, p * np.log(p), 0.0)))

# ── worker: one (K, gamma, rep) group ────────────────────────────────────────
def compute_entropy_group(args):
    """
    args: (group_key, selected_arms, K)
        group_key    : (K, gamma, rep)
        selected_arms: np.array of shape (T,) — arm chosen at each step
        K            : window size = number of arms
    Returns DataFrame with columns [step, K, gamma, rep, H, H_norm]
    """
    (K, gamma, rep), selected_arms, K_val = args

    T      = len(selected_arms)
    H      = np.zeros(T)
    H_norm = np.zeros(T)        # H / log(K)  in [0, 1]
    H_max  = np.log(K_val) if K_val > 1 else 1.0

    # rolling window of size K
    counts = np.zeros(K_val, dtype=int)

    for t in range(T):
        arm = selected_arms[t]

        # add new arm to window
        counts[arm] += 1

        # remove arm that left the window (if window full)
        if t >= K_val:
            old_arm = selected_arms[t - K_val]
            counts[old_arm] -= 1

        H[t]      = entropy_from_counts(counts)
        H_norm[t] = H[t] / H_max

    return pd.DataFrame({
        "step"  : np.arange(T),
        "K"     : K,
        "gamma" : gamma,
        "rep"   : rep,
        "H"     : H,
        "H_norm": H_norm,
    })

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    print(f"Loading {INPUT_FILE} ...")
    df = pd.read_csv(INPUT_FILE, usecols=["step","K","gamma","rep","selected_arm"])
    print(f"  {len(df):,} rows loaded in {time.time()-t_start:.1f}s")

    # build one job per (K, gamma, rep) group
    groups = df.groupby(["K","gamma","rep"])
    jobs   = []
    for (K, gamma, rep), grp in groups:
        grp_sorted    = grp.sort_values("step")
        selected_arms = grp_sorted["selected_arm"].to_numpy(dtype=int)
        jobs.append(((K, gamma, rep), selected_arms, int(K)))

    total   = len(jobs)
    n_cores = cpu_count()
    print(f"Groups  : {total}  ({n_cores} CPU core(s))")

    results = []
    with Pool(processes=n_cores) as pool:
        for done, result in enumerate(
            pool.imap_unordered(compute_entropy_group, jobs), 1
        ):
            results.append(result)
            print(f"  [{done:>4}/{total}]", end="\r")

    print()
    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["K","gamma","rep","step"]).reset_index(drop=True)

    # exp(H): effective number of arms being explored
    out["exp_H"]      = np.exp(out["H"])
    out["exp_H_norm"] = out["exp_H"] / out["K"]

    # average over replications
    avg = (
        out.groupby(["K","gamma","step"])[["H","H_norm","exp_H","exp_H_norm"]]
        .mean()
        .reset_index()
    )

    avg.to_csv(OUTPUT_FILE, index=False)
    full_file = OUTPUT_FILE.replace(".csv","_full.csv")
    out.to_csv(full_file, index=False)

    elapsed = time.time() - t_start
    print(f"\nSaved -> {OUTPUT_FILE}  (averaged over reps)")
    print(f"       -> {full_file}  (per rep)")
    print(f"  {len(avg):,} rows x {avg.shape[1]} cols  (averaged)")
    print(f"  Total time: {elapsed:.1f}s")

    print("\n--- Mean exp(H) at t=2000  (effective arms explored, max=K) ---")
    summary = (
        avg[avg.step == avg.step.max()]
        .groupby(["K","gamma"])["exp_H"]
        .mean()
        .unstack("gamma")
        .round(3)
    )
    print(summary.to_string())

    print("\n--- Mean exp(H)/K at t=2000  (1=uniform, 1/K=one arm) ---")
    summary2 = (
        avg[avg.step == avg.step.max()]
        .groupby(["K","gamma"])["exp_H_norm"]
        .mean()
        .unstack("gamma")
        .round(4)
    )
    print(summary2.to_string())

    # ── crossover: first step (after warmup) where exp_H < K/2 ───────────────
    sigma = 1.0 / np.sqrt(12)   # std of U(0,1)
    n0    = 2

    cross_rows = []
    for (K, gamma), grp in avg.groupby(["K","gamma"]):
        grp     = grp.sort_values("step")
        warm    = grp[grp["step"] >= 2*K]
        thresh  = K / 2.0
        below   = warm[warm["exp_H"] < thresh]
        t_cross = int(below["step"].iloc[0]) if len(below) else None

        # tc_th   = K / (2*gamma*sigma*np.sqrt(2*np.log(K))) - K*n0
        tc_th   = K / (2*gamma*sigma*np.sqrt(2*np.log(K))) - K*n0
        tc_th   = max(tc_th, 0.0)

        cross_rows.append({
            "K"        : K,
            "gamma"    : gamma,
            "t_cross"  : t_cross,
            "tc_theory": round(tc_th, 1),
            "ratio"    : round(t_cross/tc_th, 2) if (t_cross and tc_th > 0) else None,
        })

    cross_df = pd.DataFrame(cross_rows).sort_values(["K","gamma"])
    cross_df.to_csv("crossover_times.csv", index=False)

    print("\n--- Crossover: first step where exp(H) < K/2 ---")
    print(f"{'K':>5} {'gamma':>7} {'t_cross':>10} {'tc_theory':>12} {'ratio':>8}")
    print("-"*48)
    for _, row in cross_df.iterrows():
        print(f"{int(row.K):>5} {row.gamma:>7.2f} {str(row.t_cross):>10} "
              f"{row.tc_theory:>12} {str(row.ratio):>8}")

    print("\n=== Pivot: t_cross ===")
    print(cross_df.pivot_table(index="K", columns="gamma",
                               values="t_cross", aggfunc="first").to_string())
    print("\n=== Pivot: ratio t_cross / tc_theory ===")
    print(cross_df.pivot_table(index="K", columns="gamma",
                               values="ratio", aggfunc="first").to_string())
    print("\nSaved -> crossover_times.csv")

    # ── scatter plot: theory vs simulation ───────────────────────────────────

    plot_df    = cross_df.dropna(subset=["t_cross","tc_theory","ratio"]).copy()
    K_vals     = sorted(plot_df["K"].unique())
    gamma_vals = sorted(plot_df["gamma"].unique())
    cmap       = plt.get_cmap("viridis")
    g_palette  = cmap(np.linspace(0.15, 0.85, len(gamma_vals)))
    g_color    = {g: g_palette[i] for i, g in enumerate(gamma_vals)}
    K_sizes    = np.linspace(20, 60, len(K_vals))
    K_size     = {k: K_sizes[i] for i, k in enumerate(K_vals)}

    fig, ax = plt.subplots(figsize=(4.5, 3))

    lim = max(plot_df["tc_theory"].max(), plot_df["t_cross"].max()) * 1.15
    ax.plot([1, lim], [1, lim], "k--", lw=1)

    for _, row in plot_df.iterrows():
        ax.scatter(row.tc_theory, row.t_cross,
                   facecolors=g_color[row.gamma], edgecolors="black",
                   marker="o", linewidths=0.5,
                   s=K_size[row.K], zorder=5)
        # ax.annotate(
        #     f"\\scriptsize $k={int(row.K)}$, $\\gamma={row.gamma:.2f}$",
        #     xy=(row.tc_theory, row.t_cross),
        #     xytext=(5, 3), textcoords="offset points", fontsize=7,
        # )

    gamma_handles = [
        ax.scatter([], [], facecolors=col, edgecolors="black",
                   marker="o", linewidths=0.5, s=60)
        for col in g_color.values()
    ]
    gamma_labels = [f"$\\gamma={g:.2f}$" for g in g_color]
    k_handles = [
        ax.scatter([], [], facecolors="none", edgecolors="black",
                   marker="o", linewidths=0.5, s=size)
        for size in K_size.values()
    ]
    k_labels = [f"$k={int(k)}$" for k in K_size]
    blank_handles = [
        ax.scatter([], [], color="none", marker="o", s=0)
        for _ in range(max(0, len(k_handles) - len(gamma_handles)))
    ]

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(
        r"$t_{\rm expl}^{\rm (theory)} = "
        r"\frac{k}{2\gamma\sigma\sqrt{2\log k}} - kn_0$"
    )
    ax.set_ylabel(r"$t_{\rm expl}^{\rm (simulation)}$")
    ax.legend(
        handles=gamma_handles + blank_handles + k_handles,
        labels=gamma_labels + [""] * len(blank_handles) + k_labels,
        fontsize=7,
        ncol=2,
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(5, 1000)
    ax.set_ylim(5, 1000)

    plt.tight_layout()
    
    plt.savefig("crossover_scatter.pdf", bbox_inches="tight")
    plt.close()
    print("Saved -> crossover_scatter.pdf")