import numpy as np
from scipy.stats import beta as beta_pdf
import random
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
import seaborn as sns

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


def get_best_prob(rewards):
    nodes = list(rewards.keys())

    n_samples = len(rewards[nodes[0]])

    node_reward = np.zeros((len(nodes), n_samples))
    for node in nodes:
        node_reward[nodes.index(node)] = rewards[node]

    best_node = np.argmax(node_reward, axis=0)
    prob_best_node = dict()
    for i, node in enumerate(nodes):
        prob_best_node[node] = np.mean(best_node == i)

    return prob_best_node


class OptimalDesign:
    @staticmethod
    def get_optimal_node(
        nodes_ids, z, data, strategy, gamma=0.1
    ):
        if strategy == "oracle":
            return random.choice(list(nodes_ids))

        if strategy == "static":
            node_frequency = {
                node: len(data["nodes"][node])
                for node in nodes_ids
            }
            minimum_frequency = min(node_frequency.values())
            candidates = [
                node
                for node in nodes_ids
                if node_frequency[node] == minimum_frequency
            ]
            return random.choice(candidates)

        z_i = int(z)

        S = 5000

        G = dict()
        eig = dict()
        utility = dict()

        alphas = dict()
        betas = dict()

        for node_id in nodes_ids:
            alpha = np.ones(2)
            beta = np.ones(2)

            for trial_id, trial in data["nodes"][
                node_id
            ].items():
                if trial["y"] == True:
                    alpha[trial["z"]] += 1
                elif trial["y"] == False:
                    beta[trial["z"]] += 1

            alphas[node_id] = alpha
            betas[node_id] = beta

            alpha = alpha[:, np.newaxis]
            beta = beta[:, np.newaxis]

            phi = np.random.beta(
                alpha,
                beta,
                (2, S),
            )

            y = np.random.binomial(
                np.ones((2, S), dtype=int),
                phi,
                size=(
                    2,
                    S,
                ),
            )

            p_y_given_phi = phi * y + (1 - phi) * (1 - y)
            p_y = alpha / (alpha + beta) * y + beta / (
                alpha + beta
            ) * (1 - y)

            EIG = np.mean(
                np.log(p_y_given_phi[z_i] / p_y[z_i])
            )

            p_z = 0.5
            U = gamma * np.mean(
                p_z * phi[1]
                + (1 - p_z) * (1 - phi[0])
                - p_z * (1 - phi[1])
                - (1 - p_z) * phi[0]
            )

            U_draws = (
                p_z * phi[1]
                + (1 - p_z) * (1 - phi[0])
                - p_z * (1 - phi[1])
                - (1 - p_z) * phi[0]
            )

            G[node_id] = EIG + U
            eig[node_id] = EIG
            utility[node_id] = U_draws

        if strategy == "active_inference":
            # return sorted(
            #     list(G.keys()),
            #     key=lambda node_id: G[node_id],
            #     reverse=True,
            # )[0]
            maximum_G = max(G.values())
            candidates = [
                node
                for node in G.keys()
                if np.abs(G[node] - maximum_G) < 1e-5
            ]

            return random.choice(candidates)
        elif strategy == "greedy":
            maximum_U = max([np.mean(utility[node]) for node in utility.keys()])
            candidates = [
                node
                for node in G.keys()
                if np.abs(utility[node].mean() - maximum_U) < 1e-5
            ]
            return random.choice(candidates)
        elif strategy == "thompson_sampling":
            p_thompson = get_best_prob(utility)

            return random.choices(
                list(p_thompson.keys()),
                weights=list(p_thompson.values()),
                k=1,
            )[0]
        elif strategy == "exploration_sampling":
            p_thompson = get_best_prob(utility)
            denominator = np.sum(
                [p_thompson[node] * (1 - p_thompson[node]) for node in utility]
            )
            p_exploration = {
                node: p_thompson[node]
                * (1 - p_thompson[node])
                / denominator
                for node in utility
            }

            return random.choices(
                list(p_exploration.keys()),
                weights=list(p_exploration.values()),
                k=1,
            )[0]
        else:
            raise NotImplementedError(strategy)


class Simulator:
    def __init__(self, strategy, nodes, gamma=None):
        self.strategy = strategy
        self.nodes = nodes
        self.gamma = gamma
        self.data = {
            "participants": dict(),
            "nodes": {node_id: dict() for node_id in nodes},
        }
        self.oracle_trials = pd.read_csv(
            "output/KnowledgeTrial_oracle_treatment.csv"
        )
        self.oracle_trials.dropna(
            subset=["y"], inplace=True
        )
        self.oracle_participants = pd.read_csv(
            "output/Participant_oracle_treatment.csv",
            index_col="id",
        )
        self.oracle_participants.dropna(
            subset=["z"], inplace=True
        )
        self.participants = self.oracle_participants[
            "z"
        ].to_dict()

    def add_participant(self, participant_id):
        self.data["participants"][participant_id] = {
            "z": int(self.participants[participant_id])
        }

    def process_participant(self, participant_id):
        participant_trials = self.oracle_trials[
            self.oracle_trials["participant_id"]
            == participant_id
        ]

        if len(participant_trials) != 30:
            return

        skip_nodes = []
        n_treatments = (
            len(self.nodes)
            if self.strategy == "oracle"
            else 5
        )
        for i in range(n_treatments):
            next_node = OptimalDesign.get_optimal_node(
                set(self.nodes) - set(skip_nodes),
                self.participants[participant_id],
                self.data,
                self.strategy,
                self.gamma,
            )

            trial = self.oracle_trials[
                (
                    self.oracle_trials["participant_id"]
                    == participant_id
                )
                & (
                    self.oracle_trials["node_id"]
                    == next_node
                )
            ].iloc[0]

            self.data["nodes"][next_node][trial["id"]] = {
                "y": int(trial["y"]),
                "z": int(trial["z"]),
            }

            if i < 5:
                self.total_reward += (
                    int(trial["y"]) == int(trial["z"])
                ) * 1
                self.total_reward -= (
                    int(trial["y"]) != int(trial["z"])
                ) * 1

            skip_nodes.append(next_node)

    def simulate(self):
        self.total_reward = 0

        participants_ids = list(self.participants.keys())
        random.shuffle(participants_ids)

        for participant_id in participants_ids:
            self.add_participant(participant_id)
            self.process_participant(participant_id)

        return self.data


def beta_entropy(alpha, beta):
    import math
    from scipy.special import digamma

    if alpha <= 0 or beta <= 0:
        raise ValueError(
            "Both alpha and beta must be positive"
        )

    # Calculate log of beta function: ln(B(α,β)) = ln(Γ(α)) + ln(Γ(β)) - ln(Γ(α+β))
    log_beta = (
        math.lgamma(alpha)
        + math.lgamma(beta)
        - math.lgamma(alpha + beta)
    )

    # Calculate entropy using the formula
    entropy = (
        log_beta
        - (alpha - 1) * digamma(alpha)
        - (beta - 1) * digamma(beta)
        + (alpha + beta - 2) * digamma(alpha + beta)
    )

    return entropy


def calculate_utilities(data, nodes_ids):
    S = 50000
    utilities = {}
    entropies = {}

    for node_id in nodes_ids:
        alpha = np.ones(2)
        beta = np.ones(2)

        for trial_id, trial in data["nodes"][
            node_id
        ].items():
            if trial["y"] == True:
                alpha[trial["z"]] += 1
            elif trial["y"] == False:
                beta[trial["z"]] += 1

        entropies[node_id] = 2 * beta_entropy(1, 1)
        for z in [0, 1]:
            entropies[node_id] -= beta_entropy(
                alpha[z], beta[z]
            )

        phi = alpha / (alpha + beta)

        gamma = 1
        p_z = 0.5

        U = gamma * (
            p_z * phi[1]
            + (1 - p_z) * (1 - phi[0])
            - p_z * (1 - phi[1])
            - (1 - p_z) * phi[0]
        )

        utilities[node_id] = U

    return utilities, entropies


def calculate_strategy_accuracy(
    strategy_data, oracle_utilities, nodes_ids
):

    # Calculate utilities for both strategy and oracle
    strategy_utilities, strategy_entropies = (
        calculate_utilities(strategy_data, nodes_ids)
    )

    # Get best nodes based on mean utility
    strategy_best = max(
        strategy_utilities.keys(),
        key=lambda node: strategy_utilities[node],
    )
    oracle_best = max(
        oracle_utilities.keys(),
        key=lambda node: oracle_utilities[node],
    )

    # Calculate accuracy as 1 if they match, 0 if they don't
    accuracy = 1.0 if strategy_best == oracle_best else 0.0
    policy_reward = oracle_utilities[strategy_best]
    entropy_best_node = strategy_entropies[oracle_best]

    print(len(strategy_data["nodes"][oracle_best]))

    return (
        accuracy,
        policy_reward,
        entropy_best_node,
        strategy_utilities,
        oracle_utilities,
    )


def run_single_simulation(simulation_params):
    """
    Run a single simulation - designed to be used with multiprocessing.

    Parameters:
    - simulation_params: Tuple containing (strategy, gamma, nodes_ids, oracle_utilities, run_id)

    Returns:
    - Dictionary with simulation results
    """
    strategy, gamma, nodes_ids, oracle_utilities, run_id = (
        simulation_params
    )

    # Set a different random seed for each process to ensure different random sequences
    np.random.seed(None)
    random.seed(None)

    print(f"Starting run {run_id} for strategy {strategy}")

    # Run simulation
    simulator = Simulator(
        strategy=strategy,
        nodes=nodes_ids,
        gamma=gamma,
    )
    strategy_data = simulator.simulate()

    # Calculate accuracy
    accuracy, policy_reward, entropy_best, _, _ = (
        calculate_strategy_accuracy(
            strategy_data,
            oracle_utilities,
            nodes_ids,
        )
    )

    print(f"Completed run {run_id} for strategy {strategy}")

    return {
        "run_id": run_id,
        "accuracy": accuracy,
        "policy_reward": policy_reward,
        "entropy_best": entropy_best,
        "reward": simulator.total_reward,
    }


def evaluate_strategies_accuracy(
    strategies,
    oracle_data,
    nodes_ids,
    n_simulations=10,
    n_processes=None,
):
    if n_processes is None:
        n_processes = min(cpu_count(), n_simulations)

    print(
        f"Using {n_processes} processes for parallelization"
    )

    results = []

    oracle_utilities, oracle_entropies = (
        calculate_utilities(oracle_data, nodes_ids)
    )

    for strategy in strategies:
        if isinstance(strategy, tuple):
            strategy_name, gamma = strategy
        else:
            strategy_name, gamma = strategy, 0

        print(
            f"Running {n_simulations} simulations for strategy: {strategy_name} (gamma={gamma})"
        )

        # Prepare parameters for each simulation
        simulation_params = [
            (
                strategy_name,
                gamma,
                nodes_ids,
                oracle_utilities,
                run_id,
            )
            for run_id in range(n_simulations)
        ]

        # Run simulations in parallel
        with Pool(processes=n_processes) as pool:
            simulation_results = pool.map(
                run_single_simulation, simulation_params
            )

        # Process results
        accuracies = [
            result["accuracy"]
            for result in simulation_results
        ]
        rewards = [
            result["reward"]
            for result in simulation_results
        ]
        policy_rewards = [
            result["policy_reward"]
            for result in simulation_results
        ]
        entropies = [
            result["entropy_best"]
            for result in simulation_results
        ]

        results.append(
            {
                "strategy": strategy_name,
                "gamma": gamma,
                "accuracy": np.mean(accuracies),
                "average_reward": np.mean(rewards),
                "policy_reward": np.mean(policy_rewards),
                "entropy_best": np.mean(entropies),
                "n": len(accuracies),
                "accuracies": list(accuracies),
                "rewards": list(rewards),
                "policy_rewards": list(policy_rewards),
                "entropies": list(entropies),
            }
        )

        print(
            f"Completed {strategy_name}: accuracy={np.mean(accuracies):.3f}, reward={np.mean(rewards):.3f}"
        )

    return results


N_SIMULATIONS = 300
N_CPU = 8

import argparse
from os.path import exists


# Example usage:
def simulate(strategies, output):

    # Run oracle simulation
    print("Running oracle simulation...")
    oracle = Simulator(
        strategy="oracle", nodes=np.arange(15, 31), gamma=1
    ).simulate()



    # Evaluate strategy accuracy with parallelization
    print("Evaluating strategies for 15 treatments...")
    results_15 = evaluate_strategies_accuracy(
        strategies,
        oracle,
        np.arange(15, 31),
        n_simulations=N_SIMULATIONS,  # Increased from 1 for demonstration
        n_processes=N_CPU,  # Specify number of processes
    )

    print("Running oracle simulation for 30 treatments...")
    oracle = Simulator(
        strategy="oracle", nodes=np.arange(1, 31), gamma=1
    ).simulate()

    print("Evaluating strategies for 30 treatments...")
    results_30 = evaluate_strategies_accuracy(
        strategies,
        oracle,
        np.arange(1, 31),
        n_simulations=N_SIMULATIONS,
        n_processes=N_CPU,
    )

    results = pd.concat(
        [
            pd.DataFrame(results_15).assign(
                total_treatments=15
            ),
            pd.DataFrame(results_30).assign(
                total_treatments=30
            ),
        ]
    )
    results.to_parquet(output)
    print(
        f"Results saved to {output}"
    )


def plot_policy_regret(df, output):
    oracle_best = dict()

    oracle_15 = Simulator(
        strategy="oracle", nodes=np.arange(1, 31), gamma=1
    ).simulate()

    oracle_30 = Simulator(
        strategy="oracle", nodes=np.arange(15, 31), gamma=1
    ).simulate()

    oracle_utilities_15, oracle_entropies_15 = (
        calculate_utilities(
            oracle_15, list(oracle_15["nodes"].keys())
        )
    )
    oracle_best[15] = max(oracle_utilities_15.values())

    oracle_utilities_30, oracle_entropies_30 = (
        calculate_utilities(
            oracle_30, list(oracle_30["nodes"].keys())
        )
    )
    oracle_best[30] = max(oracle_utilities_30.values())

    fig, ax = plt.subplots(figsize=(6.4, 2.13333))
    df = df[df["total_treatments"] == 15]
    x = 0
    prev_strategy = None
    for i, row in enumerate(df.to_dict(orient="records")):
        if prev_strategy == "active_inference":
            x += 0.5
        else:
            x += 1

        prev_strategy = row["strategy"]

        policy_regret = oracle_best[
            row["total_treatments"]
        ] - np.array(
            row["policy_rewards"]
        )  # / oracle_best[row["total_treatments"]]
        print(len(policy_regret))
        sns.swarmplot(
            x=[i] * len(policy_regret),
            y=policy_regret
            + np.random.randn(len(policy_regret)) * 0.002,
            label=row["strategy"],
            s=1.5,
            edgecolor=plt.cm.tab10(i),
            ax=ax,
        )

        ax.bar(
            [i],
            [np.mean(policy_regret)],
            color=plt.cm.tab10(i),
            alpha=0.25,
        )

        mean = np.mean(policy_regret)
        low = np.quantile(policy_regret, 0.1 / 2)
        high = np.quantile(policy_regret, 1 - 0.1 / 2)
        ax.scatter([i], [mean], color=plt.cm.tab10(i))
        ax.errorbar(
            [i],
            [mean],
            ([mean - low], [high - mean]),
            color=plt.cm.tab10(i),
        )

    plt.legend()
    plt.show()
    return df


strategy_labels = {
    "static": "Even sampling (static)",
    "active_inference": "Active inference",
    "thompson_sampling": "Thompson sampling",
    "exploration_sampling": "Exploration sampling",
}

colors = {
    "static": 4,
    "active_inference": 0,
    "thompson_sampling": 9,
    "exploration_sampling": 8,
}


def plot_condition(df, plot_function, output):
    fig, ax = plt.subplots(figsize=(8.5, 3))

    labels = dict()

    labels = plot_function(
        df[df["total_treatments"] == 15], fig, ax, labels
    )

    x_mean = np.mean(list(labels.keys()))
    ax.text(
        x_mean,
        0.95 * ax.get_ylim()[1],
        "15 treatments",
        ha="center",
        va="top",
    )

    x = max(list(labels.keys()))
    ax.axvline(x + 1, color="black", ls="dashed")
    ax.text(
        x_mean + x,
        0.95 * ax.get_ylim()[1],
        "30 treatments",
        ha="center",
        va="top",
    )

    labels = plot_function(
        df[df["total_treatments"] == 30], fig, ax, labels
    )

    ax.set_xticks(
        list(labels.keys()),
        list(labels.values()),
        rotation=50,
        ha="right",
    )

    fig.savefig(output, bbox_inches="tight")


def plot_accuracy(df, fig, ax, labels):
    prev_strategy = ""

    x = max(list(labels.keys()), default=0)
    for i, row in enumerate(df.to_dict(orient="records")):
        if (
            prev_strategy == "active_inference"
            and row["strategy"] == "active_inference"
        ) or (
            "sampling" in prev_strategy
            and "sampling" in row["strategy"]
        ):
            x += 1
        else:
            x += 2

        strategy_label = strategy_labels[row["strategy"]]
        labels[x] = (
            strategy_label
            if row["strategy"] != "active_inference"
            else f"{strategy_label} ($\gamma={row['gamma']:.2f}$)"
        )

        color = colors[row["strategy"]]
        if row["strategy"] == "active_inference":
            color += [0.3, 0.2, 0.1].index(row["gamma"])

        color = plt.cm.tab20c(color)

        prev_strategy = row["strategy"]

        accuracies = row["accuracies"]
        alpha = 1 + np.sum(accuracies)
        beta = 1 + len(accuracies) - np.sum(accuracies)

        mean = np.mean(accuracies)
        low = beta_pdf(alpha, beta).ppf(0.1 / 2)
        high = beta_pdf(alpha, beta).ppf(1 - 0.1 / 2)

        print(mean, row["accuracy"])

        ax.bar(
            [x],
            [mean],
            color=color,
            alpha=0.25,
        )

        ax.scatter([x], [mean], color=color)
        ax.errorbar(
            [x],
            [mean],
            ([mean - low], [high - mean]),
            color=color,
        )

        ax.text(
            x,
            0.025,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
        )

    ax.set_ylim(0, 0.9)
    ax.set_ylabel(
        "$P(\\arg \\max_j E(r_j) = \\arg \\max_j E(r_j)_{oracle}) $"
    )

    return labels


def plot_information_gain(df, fig, ax, labels):
    prev_strategy = ""

    x = max(list(labels.keys()), default=0)
    for i, row in enumerate(df.to_dict(orient="records")):
        if (
            prev_strategy == "active_inference"
            and row["strategy"] == "active_inference"
        ) or (
            "sampling" in prev_strategy
            and "sampling" in row["strategy"]
        ):
            x += 1
        else:
            x += 2

        strategy_label = strategy_labels[row["strategy"]]
        labels[x] = (
            strategy_label
            if row["strategy"] != "active_inference"
            else f"{strategy_label} ($\gamma={row['gamma']:.2f}$)"
        )

        color = colors[row["strategy"]]
        if row["strategy"] == "active_inference":
            color += [0.3, 0.2, 0.1].index(row["gamma"])

        color = plt.cm.tab20c(color)

        prev_strategy = row["strategy"]

        entropies = row["entropies"]

        mean = np.mean(entropies)
        low = np.quantile(entropies, q=0.1 / 2)
        high = np.quantile(entropies, q=1 - 0.1 / 2)

        print(mean, row["accuracy"])

        violin = ax.violinplot(
            dataset=entropies,
            positions=[x],
            showextrema=False
        )

        print(violin.keys())

        for pc in violin["bodies"]:
            pc.set_facecolor(color)
            pc.set_edgecolor(color)
            pc.set_alpha(0.25)

        ax.scatter([x], [mean], color=color)
        ax.errorbar(
            [x],
            [mean],
            ([mean - low], [high - mean]),
            color=color,
            capsize=4
        )

    ax.set_ylabel("$IG(\\hat{j})$")
    ax.set_ylim(0.25, 4)

    return labels


def plot(df):
    plot_condition(
        df, plot_accuracy, "output/rl_accuracy.pdf"
    )
    plot_condition(
        df, plot_information_gain, "output/rl_ig.pdf"
    )
    plot_policy_regret(df, "output/rl_policy_regret.pdf")


if __name__ == "__main__":
    strategies = [
        "exploration_sampling"
    ]
    output = "output/rl_exploration_sampling.parquet"
    simulate(strategies, output)

    if exists("output/rl_large_randomized.parquet"):
        results = pd.read_parquet(
            "output/rl_large_randomized.parquet"
        )
        plot(results)
    else:
        strategies = [
            "static",
            ("active_inference", 0.1),
            ("active_inference", 0.2),
            ("active_inference", 0.3),
            "thompson_sampling",
            "exploration_sampling",
        ]
        output="output/rl_large_randomized.parquet"
        simulate(strategies, output)
