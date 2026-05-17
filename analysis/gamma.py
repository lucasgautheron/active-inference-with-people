"""
Active inference bandit simulation.
Exports per-step G, EIG, U for every arm + selected arm to CSV.

Configurations: all combinations of K and gamma.
True utilities: theta_j ~ U(0, 1).
Prior: Beta(1, 1) for each arm.
T = 2000 steps per configuration.
"""

import numpy as np
import pandas as pd
from scipy.special import digamma
from itertools import product
from multiprocessing import Pool, cpu_count
import time

# ── parameters ────────────────────────────────────────────────────────────────
K_VALUES     = [5, 10, 15, 30, 60]
GAMMA_VALUES = [0.05, 0.1, 0.2, 0.3]
T            = 1000
N_REP        = 30
SEED         = 42
OUTPUT_FILE  = "output/gamma_active_inference_sims.csv"

rng = np.random.default_rng(SEED)

# ── exact analytical EIG ──────────────────────────────────────────────────────
def eig_exact(alpha, beta):
    """
    Exact EIG for Beta(alpha, beta) prior with Bernoulli likelihood.
    EIG = H(y) - E_theta[H(y|theta)]
        = [-mu*log(mu) - (1-mu)*log(1-mu)]
          + mu*psi(alpha+1) + (1-mu)*psi(beta+1) - psi(n+1)
    """
    n   = alpha + beta
    mu  = alpha / n
    # guard against boundary
    if mu <= 0 or mu >= 1:
        return 0.0
    h_marginal = -mu*np.log(mu) - (1-mu)*np.log(1-mu)
    exp_cond   = (digamma(n+1)
                  - mu*digamma(alpha+1)
                  - (1-mu)*digamma(beta+1))
    return float(h_marginal - exp_cond)

# vectorised over arms
def eig_all(alphas, betas):
    return np.array([eig_exact(alphas[j], betas[j]) for j in range(len(alphas))])

# ── single run ────────────────────────────────────────────────────────────────
def run_simulation(K, gamma, T, theta, rng):
    """
    Run one episode of T steps.

    Returns a list of dicts, one per step, with fields:
        step, K, gamma, rep, theta_j (per arm), selected_arm,
        eig_j, u_j, g_j (per arm)
    """
    alpha = np.ones(K)
    beta  = np.ones(K)
    rows  = []

    for t in range(T):
        mu  = alpha / (alpha + beta)
        eig = eig_all(alpha, beta)
        u   = gamma * mu                # expected utility (scaled)
        g   = eig + u                   # EFE (to maximise)

        j_star = int(np.argmax(g))

        # build row dict
        row = {
            "step"        : t,
            "K"           : K,
            "gamma"       : gamma,
            "selected_arm": j_star,
        }
        for j in range(K):
            row[f"theta_{j}"] = float(theta[j])
            row[f"eig_{j}"]   = float(eig[j])
            row[f"u_{j}"]     = float(u[j])
            row[f"g_{j}"]     = float(g[j])
            row[f"n_{j}"]     = float(alpha[j] + beta[j])
            row[f"mu_{j}"]    = float(mu[j])

        rows.append(row)

        # observe and update
        y = rng.binomial(1, theta[j_star])
        alpha[j_star] += y
        beta[j_star]  += 1 - y

    return rows

# ── worker function (must be top-level for pickle) ────────────────────────────
def worker(args):
    """One run: (K, gamma, rep, seed) → list of row dicts."""
    K, gamma, rep, seed = args
    local_rng = np.random.default_rng(seed)
    theta     = local_rng.uniform(0, 1, size=K)
    rows      = run_simulation(K, gamma, T, theta, local_rng)
    for row in rows:
        row["rep"] = rep
    return rows

if __name__ == '__main__':
    # ── build job list ────────────────────────────────────────────────────────────
    configs  = list(product(K_VALUES, GAMMA_VALUES))
    jobs     = [
        (K, gamma, rep, SEED + idx * N_REP + rep)
        for idx, (K, gamma) in enumerate(configs)
        for rep in range(N_REP)
    ]
    total    = len(jobs)
    n_cores  = 4

    print(f"Configurations : {len(configs)}  (K × gamma)")
    print(f"Replicates     : {N_REP}")
    print(f"Steps each     : {T}")
    print(f"Total runs     : {total}")
    print(f"CPU cores      : {n_cores}")
    print(f"Output file    : {OUTPUT_FILE}")
    print("-" * 55)

    # ── parallel execution ────────────────────────────────────────────────────────
    t_start  = time.time()
    all_rows = []

    with Pool(processes=n_cores) as pool:
        for done, rows in enumerate(pool.imap_unordered(worker, jobs), 1):
            all_rows.extend(rows)
            elapsed = time.time() - t_start
            eta     = elapsed / done * (total - done)
            print(f"  [{done:>4}/{total}]  {elapsed:.1f}s elapsed  "
                f"ETA {eta:.0f}s", end="\r")

    print()
    print(f"\nTotal rows: {len(all_rows):,}")

    # ── assemble dataframe & export ───────────────────────────────────────────────
    df = pd.DataFrame(all_rows)

    # reorder columns: metadata first, then per-arm fields grouped by arm
    meta_cols = ["step","K","gamma","rep","selected_arm"]
    K_max     = max(K_VALUES)
    arm_cols  = []
    for j in range(K_max):
        for field in ["theta","eig","u","g","n","mu"]:
            c = f"{field}_{j}"
            if c in df.columns:
                arm_cols.append(c)

    ordered = meta_cols + arm_cols
    df      = df[[c for c in ordered if c in df.columns]]

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")
    print(f"File size: {df.memory_usage(deep=True).sum()/1e6:.1f} MB (in memory)")
    print(f"Total time: {time.time()-t_start:.1f}s")

    # ── quick sanity check ────────────────────────────────────────────────────────
    print("\n--- Sanity check ---")
    for K, gamma in [(2, 0.1), (15, 0.2)]:
        sub = df[(df.K==K) & (df.gamma==gamma) & (df.rep==0)]
        print(f"K={K}, gamma={gamma}: {len(sub)} rows, "
            f"selected_arm range [{sub.selected_arm.min()}, "
            f"{sub.selected_arm.max()}], "
            f"mean EIG(arm0) = {sub.eig_0.mean():.4f}")