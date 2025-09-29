import numpy as np
import random
import pandas as pd
from multiprocessing import Pool, cpu_count

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
    def get_optimal_node(nodes_ids, data, strategy, gamma=0.1):
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

        S = 5000

        G = dict()
        eig = dict()
        utility = dict()

        alphas = dict()
        betas = dict()

        for node_id in nodes_ids:
            alpha = 1.0
            beta = 1.0

            for trial_id, trial in data["nodes"][node_id].items():
                if trial["y"] == True:
                    alpha += 1
                elif trial["y"] == False:
                    beta += 1

            alphas[node_id] = alpha
            betas[node_id] = beta

            phi = np.random.beta(alpha, beta, S)
            y = np.random.binomial(np.ones(S, dtype=int), phi)

            p_y_given_phi = phi * y + (1 - phi) * (1 - y)
            p_y = alpha / (alpha + beta) * y + beta / (alpha + beta) * (1 - y)

            if strategy == "active_inference":
                EIG = np.mean(np.log(p_y_given_phi / p_y))
                U = gamma * np.mean(phi)
                G[node_id] = EIG + U
                eig[node_id] = EIG

            utility[node_id] = phi

        if strategy == "active_inference":
            maximum_G = max(G.values())
            candidates = [
                node
                for node in G.keys()
                if np.abs(G[node] - maximum_G) < 1e-5
            ]
            return random.choice(candidates)
        elif strategy == "greedy":
            maximum_U = max(
                [np.mean(utility[node]) for node in utility.keys()]
            )
            candidates = [
                node
                for node in utility.keys()
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
                [
                    p_thompson[node] * (1 - p_thompson[node])
                    for node in utility
                ]
            )
            if denominator == 0:
                p_exploration = p_thompson
            else:
                p_exploration = {
                    node: p_thompson[node] * (1 - p_thompson[node]) / denominator
                    for node in utility
                }

            return random.choices(
                list(p_exploration.keys()),
                weights=list(p_exploration.values()),
                k=1,
            )[0]
        else:
            raise NotImplementedError(strategy)


def calculate_current_utilities(data, nodes_ids):
    """Calculate current utility estimates based on observed data"""
    utilities = {}

    for node_id in nodes_ids:
        alpha = 1.0
        beta = 1.0

        for trial_id, trial in data["nodes"][node_id].items():
            if trial["y"] == True:
                alpha += 1
            elif trial["y"] == False:
                beta += 1

        phi = alpha / (alpha + beta)
        utilities[node_id] = phi

    return utilities


class MABSimulator:
    def __init__(
        self,
        strategy,
        nodes,
        gamma=None,
        n_participants=400,
        n_trials_per_participant=1,
        true_thetas=None,
    ):
        self.strategy = strategy
        self.nodes = nodes
        self.gamma = gamma
        self.n_participants = n_participants
        self.n_trials_per_participant = n_trials_per_participant
        self.data = {
            "participants": dict(),
            "nodes": {node_id: dict() for node_id in nodes},
        }
        self.total_reward = 0
        self.trial_counter = 0
        self.iteration_data = []
        self.true_thetas = true_thetas

    def add_participant(self, participant_id):
        # No longer need to assign treatment condition
        self.data["participants"][participant_id] = {}

    def generate_outcome(self, node_id):
        """Generate outcome y based on true theta"""
        theta = self.true_thetas[node_id]
        return np.random.binomial(1, theta)

    def get_current_best_treatment(self):
        """Get the currently believed best treatment based on utility estimates"""
        if self.trial_counter == 0:
            return None

        utilities = calculate_current_utilities(self.data, self.nodes)
        if not utilities:
            return None

        return max(utilities.keys(), key=lambda node: utilities[node])

    def get_true_best_treatment(self):
        """Get the actual best treatment based on true parameters"""
        return max(self.nodes, key=lambda node: self.true_thetas[node])

    def get_true_utility(self, node_id):
        """Get the true utility of a specific treatment"""
        return self.true_thetas[node_id]

    def process_participant(self, participant_id):
        skip_nodes = []
        n_treatments = (
            len(self.nodes)
            if self.strategy == "oracle"
            else self.n_trials_per_participant
        )

        for i in range(n_treatments):
            next_node = OptimalDesign.get_optimal_node(
                set(self.nodes) - set(skip_nodes),
                self.data,
                self.strategy,
                self.gamma,
            )

            # Generate outcome based on true MAB parameters
            y = self.generate_outcome(next_node)

            # Store the trial data
            self.data["nodes"][next_node][self.trial_counter] = {
                "y": int(y),
            }

            # Calculate reward
            self.total_reward += int(y)

            # Record iteration data
            current_best = self.get_current_best_treatment()
            true_best = self.get_true_best_treatment()

            current_best_utility = (
                self.get_true_utility(current_best)
                if current_best is not None
                else 0
            )
            true_best_utility = self.get_true_utility(true_best)

            self.iteration_data.append(
                {
                    "run_id": getattr(self, "run_id", 0),
                    "strategy": self.strategy,
                    "gamma": self.gamma,
                    "iteration": self.trial_counter,
                    "participant_id": participant_id,
                    "treatment_within_participant": i,
                    "selected_treatment": next_node,
                    "outcome_y": int(y),
                    "cumulative_reward": self.total_reward,
                    "current_best_treatment": current_best,
                    "true_best_treatment": true_best,
                    "true_utility_current_best": current_best_utility,
                    "true_utility_true_best": true_best_utility,
                    "true_utility_treatment": self.get_true_utility(next_node),
                    "total_treatments": len(self.nodes),
                }
            )

            skip_nodes.append(next_node)
            self.trial_counter += 1

    def simulate(self, run_id=0):
        self.run_id = run_id

        # Generate true_thetas if not already provided
        if self.true_thetas is None:
            self.true_thetas = {
                node_id: np.random.beta(2, 2) for node_id in self.nodes
            }

        self.total_reward = 0
        self.trial_counter = 0
        self.iteration_data = []

        for participant_id in range(self.n_participants):
            self.add_participant(participant_id)
            self.process_participant(participant_id)

        return self.data, self.true_thetas, self.iteration_data


def beta_entropy(alpha, beta):
    import math
    from scipy.special import digamma

    if alpha <= 0 or beta <= 0:
        raise ValueError("Both alpha and beta must be positive")

    log_beta = (
        math.lgamma(alpha)
        + math.lgamma(beta)
        - math.lgamma(alpha + beta)
    )

    entropy = (
        log_beta
        - (alpha - 1) * digamma(alpha)
        - (beta - 1) * digamma(beta)
        + (alpha + beta - 2) * digamma(alpha + beta)
    )

    return entropy


def run_single_simulation(simulation_params):
    """
    Run a single simulation - designed to be used with multiprocessing.
    """
    strategy, gamma, nodes_ids, run_id, true_thetas = simulation_params

    # Set a different random seed for each process
    np.random.seed(None)
    random.seed(None)

    print(f"Starting run {run_id} for strategy {strategy}")

    simulator = MABSimulator(
        strategy=strategy,
        nodes=nodes_ids,
        gamma=gamma,
        true_thetas=true_thetas,
    )
    _, _, iteration_data = simulator.simulate(run_id)

    print(f"Completed run {run_id} for strategy {strategy}")

    return iteration_data


def evaluate_strategies(
    strategies,
    nodes_ids,
    n_simulations=10,
    n_processes=None,
):
    if n_processes is None:
        n_processes = min(cpu_count(), n_simulations)

    print(f"Using {n_processes} processes for parallelization")

    all_iteration_data = []

    # Generate the true MABs once for all strategies
    print(f"Generating {n_simulations} random MABs...")
    mab_instances = []
    for run_id in range(n_simulations):
        np.random.seed(run_id + 1000)
        true_thetas = {
            node_id: np.random.beta(2, 2) for node_id in nodes_ids
        }
        mab_instances.append(true_thetas)

    np.random.seed(None)

    for strategy in strategies:
        if isinstance(strategy, tuple):
            strategy_name, gamma = strategy
        else:
            strategy_name, gamma = strategy, 0

        print(
            f"Running {n_simulations} simulations for strategy: {strategy_name} (gamma={gamma})"
        )

        simulation_params = [
            (
                strategy_name,
                gamma,
                nodes_ids,
                run_id,
                mab_instances[run_id],
            )
            for run_id in range(n_simulations)
        ]

        with Pool(processes=n_processes) as pool:
            simulation_results = pool.map(
                run_single_simulation, simulation_params
            )

        for iteration_data in simulation_results:
            all_iteration_data.extend(iteration_data)

        print(f"Completed {strategy_name}")

    return all_iteration_data


N_SIMULATIONS = 300
N_CPU = 8


def simulate(strategies, output_iterations):
    print("Evaluating strategies for 5 treatments...")
    iterations_5 = evaluate_strategies(
        strategies,
        np.arange(0, 5),
        n_simulations=N_SIMULATIONS,
        n_processes=N_CPU,
    )

    print("Evaluating strategies for 10 treatments...")
    iterations_10 = evaluate_strategies(
        strategies,
        np.arange(0, 10),
        n_simulations=N_SIMULATIONS,
        n_processes=N_CPU,
    )

    print("Evaluating strategies for 20 treatments...")
    iterations_30 = evaluate_strategies(
        strategies,
        np.arange(0, 30),
        n_simulations=N_SIMULATIONS,
        n_processes=N_CPU,
    )

    all_iterations = pd.DataFrame(
        iterations_5 + iterations_10 + iterations_30
    )
    all_iterations.to_csv(output_iterations, index=False)
    print(f"Iteration-level data saved to {output_iterations}")


strategy_labels = {
    "static": "Even sampling",
    "active_inference": "Active inference",
    "thompson_sampling": "Thompson sampling",
    "exploration_sampling": "Exploration sampling",
    "greedy": "Greedy",
}

colors = {
    "static": 4,
    "active_inference": 0,
    "thompson_sampling": 9,
    "exploration_sampling": 8,
    "greedy": 12,
}

linestyles = {
    "static": "dotted",
    "active_inference": "-",
    "thompson_sampling": "dashed",
    "exploration_sampling": "dashed",
    "greedy": "dotted",
}


def plot_combined_regret(df):
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(
        4, 3, figsize=(12.5 * 0.7, 10 * 0.7 * 4/3), sharey="row"
    )

    # Treatment counts to plot
    treatment_counts = [5, 10, 30]

    for col, n_treatments in enumerate(treatment_counts):
        ax = axes[0, col]
        df_subset = df[
            df["total_treatments"] == n_treatments
        ]

        for (strategy, gamma), runs in df_subset.groupby(
            ["strategy", "gamma"]
        ):
            runs["correct"] = runs["current_best_treatment"] == runs["true_best_treatment"]
            iterations = runs.groupby("iteration").agg(
                correct=(
                    "correct",
                    "mean",
                ),
            )

            strategy_label = strategy_labels[strategy]
            label = (
                strategy_label
                if strategy != "active_inference"
                else f"{strategy_label} ($\gamma={gamma:.1f}$)"
            )

            color = colors[strategy]
            if strategy == "active_inference":
                color += [0.3, 0.2, 0.1].index(gamma)

            color = plt.cm.tab20c(color)

            indices = np.arange(1, len(iterations), 10)
            ax.plot(
                indices,
                iterations["correct"].values[indices],
                label=label,
                color=color,
                ls=linestyles[strategy],
            )

        ax.set_ylim(0, 1)
        ax.set_title(f"{n_treatments} treatments")
        # ax.set_xlabel("Iteration")
        if col == 0:
            ax.set_ylabel("Best arm identification")

    # Top row: Policy regret
    for col, n_treatments in enumerate(treatment_counts):
        ax = axes[1, col]
        df_subset = df[
            df["total_treatments"] == n_treatments
        ]

        for (strategy, gamma), runs in df_subset.groupby(
            ["strategy", "gamma"]
        ):
            iterations = runs.groupby("iteration").agg(
                true_utility_current_best=(
                    "true_utility_current_best",
                    "mean",
                ),
                true_utility_true_best=(
                    "true_utility_true_best",
                    "mean",
                ),
            )
            iterations["policy_regret"] = (
                iterations["true_utility_true_best"]
                - iterations["true_utility_current_best"]
            )

            strategy_label = strategy_labels[strategy]
            label = (
                strategy_label
                if strategy != "active_inference"
                else f"{strategy_label} ($\gamma={gamma:.1f}$)"
            )

            color = colors[strategy]
            if strategy == "active_inference":
                color += [0.3, 0.2, 0.1].index(gamma)

            color = plt.cm.tab20c(color)

            indices = np.arange(1, len(iterations), 10)
            ax.plot(
                indices,
                iterations["policy_regret"].values[indices],
                label=label,
                color=color,
                ls=linestyles[strategy],
            )

        ax.set_ylim(0, 0.5)
        ax.set_title(f"{n_treatments} treatments")
        # ax.set_xlabel("Iteration")
        if col == 0:
            ax.set_ylabel("Policy regret")

    # Bottom row: Sample regret (cumulative)
    for col, n_treatments in enumerate(treatment_counts):
        ax = axes[2, col]
        df_subset = df[
            df["total_treatments"] == n_treatments
        ]

        for (strategy, gamma), runs in df_subset.groupby(
            ["strategy", "gamma"]
        ):
            runs["regret"] = (
                runs["true_utility_true_best"]
                - runs["true_utility_treatment"]
            )
            runs["regret"] = runs.groupby("run_id")[
                "regret"
            ].cumsum() / ((runs["iteration"] + 1))

            iterations = runs.groupby("iteration").agg(
                regret=("regret", "mean")
            )

            strategy_label = strategy_labels[strategy]
            label = (
                strategy_label
                if strategy != "active_inference"
                else f"{strategy_label} ($\gamma={gamma:.1f}$)"
            )

            color = colors[strategy]
            if strategy == "active_inference":
                color += [0.3, 0.2, 0.1].index(gamma)

            color = plt.cm.tab20c(color)

            ax.plot(
                iterations["regret"],
                label=label,
                color=color,
                ls=linestyles[strategy],
            )

        ax.set_title(f"{n_treatments} treatments")
        if col == 0:
            ax.set_ylabel("Average regret")

    for col, n_treatments in enumerate(treatment_counts):
        ax = axes[3, col]
        df_subset = df[
            df["total_treatments"] == n_treatments
        ]

        for (strategy, gamma), runs in df_subset.groupby(
            ["strategy", "gamma"]
        ):
            runs["exploit"] = (
                runs["selected_treatment"]
                == runs["current_best_treatment"]
            )

            iterations = runs.groupby("iteration").agg(
                exploit=("exploit", "mean")
            )

            strategy_label = strategy_labels[strategy]
            label = (
                strategy_label
                if strategy != "active_inference"
                else f"{strategy_label} ($\gamma={gamma:.1f}$)"
            )

            color = colors[strategy]
            if strategy == "active_inference":
                color += [0.3, 0.2, 0.1].index(gamma)

            color = plt.cm.tab20c(color)

            indices = np.arange(2, len(iterations), 10)
            ax.plot(
                indices,
                (
                    iterations["exploit"].values[indices]
                    if strategy != "greedy"
                    else np.ones(len(indices))
                ),
                label=label,
                color=color,
                ls=linestyles[strategy],
            )

        ax.set_title(f"{n_treatments} treatments")
        ax.set_xlabel("Iteration")
        if col == 0:
            ax.set_ylabel("Exploitation probability")

    # Organize legend entries by desired grouping
    handles, labels = axes[0, 0].get_legend_handles_labels()

    # Create organized lists for each column
    col1_entries = []  # Even sampling and Greedy
    col2_entries = []  # Active inference (all gamma values)
    col3_entries = []  # Thompson and Exploration sampling

    # Sort entries into appropriate columns
    for handle, label in zip(handles, labels):
        if (
            "even" in label.lower()
            or "greedy" in label.lower()
        ):
            col1_entries.append((handle, label))
        elif "active inference" in label.lower():
            col2_entries.append((handle, label))
        elif (
            "thompson" in label.lower()
            or "exploration" in label.lower()
        ):
            col3_entries.append((handle, label))

    # Combine in desired order: col1, then col2, then col3
    organized_handles = []
    organized_labels = []

    for handle, label in col1_entries:
        organized_handles.append(handle)
        organized_labels.append(label)

    organized_labels.append("")
    organized_handles.append(
        matplotlib.patches.Rectangle(
            (0, 0),
            1,
            1,
            fill=False,
            edgecolor="none",
            visible=False,
        )
    )

    for handle, label in col2_entries:
        organized_handles.append(handle)
        organized_labels.append(label)

    for handle, label in col3_entries:
        organized_handles.append(handle)
        organized_labels.append(label)

    # Add legend with organized entries
    fig.legend(
        organized_handles,
        organized_labels,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        frameon=False,
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for legend
    plt.savefig(
        "output/systematic_active_inference_simple.pdf",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    strategies = [
        "static",
        ("active_inference", 0.1),
        ("active_inference", 0.2),
        ("active_inference", 0.3),
        "thompson_sampling",
        "exploration_sampling",
        "greedy",
    ]
    output_iterations = "output/mab_iterations_400_simple.csv"
    # simulate(strategies, output_iterations)
    df = pd.read_csv(output_iterations)
    plot_combined_regret(df)
    # Uncomment the line below to run simulations instead
