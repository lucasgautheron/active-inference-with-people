import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


import torch
import pyro
import pyro.distributions as dist
import numpy as np
from torch.distributions.constraints import positive
from pyro.contrib.oed.eig import marginal_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from scipy.stats import norm, lognorm, beta as beta_dist


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
    def __init__(self):

        # Priors parameters
        self.prior_mean_theta = torch.tensor(0.0)
        self.prior_sd_theta = torch.tensor(2.0)
        self.prior_mean_difficulty = torch.tensor(0.0)
        self.prior_sd_difficulty = torch.tensor(1.0)
        self.prior_mean_intercept = torch.tensor(0.0)
        self.prior_sd_intercept = torch.tensor(1.0)

        # Duration parameter priors (exponential with rate=1)
        self.prior_tau_rate = torch.tensor(1.0)

        self.theta_means = torch.empty(0)
        self.theta_sds = torch.empty(0)
        self.difficulty_means = torch.empty(0)
        self.difficulty_sds = torch.empty(0)
        self.intercept_mean = torch.tensor(0.0)
        self.intercept_sd = torch.tensor(0.0)

        # Duration parameters (log-normal guide parameters)
        self.log_tau_mus = torch.empty(0)
        self.log_tau_sds = torch.empty(0)

        # EIG computation parameters
        self.num_steps = 400
        self.start_lr = 0.1
        self.end_lr = 0.001

    def _make_design_model(self, target_participant):
        """Create a model for a specific participant
        that takes item indices as design"""

        def model(design):
            with pyro.plate_stack(
                "plate", design.shape[:-1]
            ):
                # Sample ability parameter for the target participant
                theta = pyro.sample(
                    "theta",
                    dist.Normal(
                        self.theta_means[
                            target_participant
                        ],
                        self.theta_sds[target_participant],
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

                intercept = pyro.sample(
                    "intercept",
                    dist.Normal(
                        self.intercept_mean,
                        self.intercept_sd,
                    ),
                ).unsqueeze(-1)

                # Sample duration parameter for this specific item
                item_tau = pyro.sample(
                    "tau",
                    dist.LogNormal(
                        self.log_tau_mus[item_idx],
                        self.log_tau_sds[item_idx],
                    ),
                ).unsqueeze(-1)

                logit_p = (theta - difficulties) + intercept

                y = pyro.sample(
                    "y",
                    dist.Bernoulli(
                        logits=logit_p,
                    ).to_event(1),
                )

                # Sample duration from exponential distribution
                x = pyro.sample(
                    "x",
                    dist.Exponential(item_tau).to_event(1),
                )

                return y

        return model

    def _model(self, participants, items, durations=None):
        """Model of the data-generating process"""
        # Sample ability parameters
        # for all participants: theta_i ~ N(0, 2)
        thetas = pyro.sample(
            "thetas",
            dist.Normal(
                self.prior_mean_theta,
                self.prior_sd_theta,
            )
            .expand(
                [self.num_participants],
            )
            .to_event(1),
        )

        # Sample difficulty parameters
        # for all potential items
        difficulties = pyro.sample(
            "difficulties",
            dist.Normal(
                self.prior_mean_difficulty,
                self.prior_sd_difficulty,
            )
            .expand(
                [self.num_items],
            )
            .to_event(1),
        )

        # Sample intercept parameter
        intercept = pyro.sample(
            "intercept",
            dist.Normal(
                self.prior_mean_intercept,
                self.prior_sd_intercept,
            ),
        )

        # Sample duration parameters for all items
        taus = pyro.sample(
            "taus",
            dist.Exponential(self.prior_tau_rate)
            .expand([self.num_items])
            .to_event(1),
        )

        selected_thetas = thetas[participants.long()]
        selected_difficulties = difficulties[items.long()]
        selected_taus = taus[items.long()]

        # Logistic regression model with intercept
        logit_p = (
            selected_thetas - selected_difficulties
        ) + intercept
        y = pyro.sample(
            "y",
            dist.Bernoulli(
                logits=logit_p,
            ).to_event(1),
        )

        # Duration model - exponential distribution with item-specific rates
        if durations is not None:
            x = pyro.sample(
                "x",
                dist.Exponential(selected_taus).to_event(1),
            )

        return y

    def _guide(self, participants, items, durations=None):
        """Guide for multiple participants
        with hierarchical theta structure"""

        # Guide for thetas
        # (means and sds)
        theta_means = pyro.param(
            "theta_means",
            torch.full(
                [self.num_participants],
                self.prior_mean_theta,
            ),
        )
        theta_sds = pyro.param(
            "theta_sds",
            torch.full(
                [self.num_participants],
                self.prior_sd_theta,
            ),
            constraint=positive,
        )
        pyro.sample(
            "thetas",
            dist.Normal(
                theta_means,
                theta_sds,
            ).to_event(1),
        )

        # Guide for difficulties
        mean_difficulties = pyro.param(
            "mean_difficulties",
            self.prior_mean_difficulty.expand(
                [self.num_items],
            ).clone(),
        )
        sd_difficulties = pyro.param(
            "sd_difficulties",
            self.prior_sd_difficulty.expand(
                [self.num_items],
            ).clone(),
            constraint=positive,
        )
        pyro.sample(
            "difficulties",
            dist.Normal(
                mean_difficulties,
                sd_difficulties,
            ).to_event(1),
        )

        # Guide for intercept
        mean_intercept = pyro.param(
            "mean_intercept",
            self.prior_mean_intercept.clone(),
        )
        sd_intercept = pyro.param(
            "sd_intercept",
            self.prior_sd_intercept.clone(),
            constraint=positive,
        )
        pyro.sample(
            "intercept",
            dist.Normal(
                mean_intercept,
                sd_intercept,
            ),
        )

        # Guide for duration parameters (log-normal for each item)
        log_tau_mu_params = pyro.param(
            "log_tau_mu_params",
            torch.zeros(self.num_items),
        )
        log_tau_sd_params = pyro.param(
            "log_tau_sd_params",
            torch.ones(self.num_items),
            constraint=positive,
        )
        pyro.sample(
            "taus",
            dist.LogNormal(
                log_tau_mu_params,
                log_tau_sd_params,
            ).to_event(1),
        )

    def _marginal_guide(
        self,
        design,
        observation_labels,
        target_labels,
    ):
        """Guide for marginal_eig"""
        q_logit = pyro.param(
            "q_logit",
            torch.zeros(design.shape[-2:]),
        )
        pyro.sample(
            "y",
            dist.Bernoulli(logits=q_logit).to_event(
                1,
            ),
        )

    def init_parameters(self, num_participants, num_items):
        self.num_participants = num_participants
        self.num_items = num_items

        self.theta_means = torch.full(
            [num_participants], self.prior_mean_theta
        )
        self.theta_sds = torch.full(
            [num_participants], self.prior_sd_theta
        )
        self.difficulty_means = torch.full(
            [num_items], self.prior_mean_difficulty
        )
        self.difficulty_sds = torch.full(
            [num_items], self.prior_sd_difficulty
        )
        self.intercept_mean = torch.tensor(
            self.prior_mean_intercept
        )
        self.intercept_sd = torch.tensor(
            self.prior_sd_intercept
        )

        # Initialize duration parameters (log-normal guide parameters for each item)
        self.log_tau_mus = torch.zeros(num_items)
        self.log_tau_sds = torch.ones(num_items)

    def update_posterior(self, data):
        """Update posterior beliefs based on all collected experimental data"""

        participants = []
        items = []
        responses = []
        durations = []

        for node_id, trials in data["nodes"].items():
            for trial_id, trial_data in trials.items():
                participants.append(
                    trial_data["participant_id"]
                )
                items.append(node_id)
                responses.append(float(trial_data["y"]))
                durations.append(
                    float(trial_data["x"])
                )  # Add duration data

        print("durations:")
        print(np.mean(durations))

        self.participant_index = {
            participant: idx
            for idx, participant in enumerate(
                data["participants"]
            )
        }
        self.item_index = {
            item: idx
            for idx, item in enumerate(data["nodes"].keys())
        }

        participants = torch.tensor(
            [
                self.participant_index[participant]
                for participant in participants
            ],
            dtype=torch.long,
        )
        items = torch.tensor(
            [self.item_index[item] for item in items],
            dtype=torch.long,
        )
        responses = torch.tensor(responses)
        durations = torch.tensor(durations)

        # Initialize parameters with correct sizes
        self.init_parameters(
            len(self.participant_index),
            len(self.item_index),
        )

        print(self.participant_index, self.item_index)

        pyro.clear_param_store()

        # The statistical model conditioned on all prior responses and durations
        conditioned_model = pyro.condition(
            self._model, {"y": responses, "x": durations}
        )

        # Instantiate the stochastic variational inference
        svi = SVI(
            conditioned_model,
            self._guide,
            Adam({"lr": 0.02}),
            loss=Trace_ELBO(),
        )

        # Fit the model
        for i in range(self.num_steps):
            elbo = svi.step(participants, items, durations)
            if i % 100 == 0:
                print(f"  Iteration {i}, ELBO: {elbo:.3f}")

        # Extract parameters
        self.theta_means = (
            pyro.param("theta_means").detach().clone()
        )
        self.theta_sds = (
            pyro.param("theta_sds").detach().clone()
        )
        self.difficulty_means = (
            pyro.param("mean_difficulties").detach().clone()
        )
        self.difficulty_sds = (
            pyro.param("sd_difficulties").detach().clone()
        )
        self.intercept_mean = (
            pyro.param("mean_intercept").detach().clone()
        )
        self.intercept_sd = (
            pyro.param("sd_intercept").detach().clone()
        )

        # Extract duration parameters
        self.log_tau_mus = (
            pyro.param("log_tau_mu_params").detach().clone()
        )
        self.log_tau_sds = (
            pyro.param("log_tau_sd_params").detach().clone()
        )

        print("Posterior update completed")

    def get_optimal_node(
        self, candidates, participant, data
    ):
        # Update posterior with current data
        self.update_posterior(data)

        # Create design model for this participant
        pyro.clear_param_store()
        design_model = self._make_design_model(
            self.participant_index[participant]
        )

        # Candidate designs
        candidate_designs = torch.tensor(
            [self.item_index[item] for item in candidates],
            dtype=torch.float,
        ).unsqueeze(-1)

        optimizer = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": self.start_lr},
                "gamma": (self.end_lr / self.start_lr)
                ** (1 / self.num_steps),
            }
        )

        # Compute Expected Information Gain for each candidate item
        eig = marginal_eig(
            design_model,
            candidate_designs,
            "y",
            ["theta", "difficulties", "intercept"],
            num_samples=100,
            num_steps=self.num_steps,
            guide=self._marginal_guide,
            optim=optimizer,
            final_num_samples=10000,
        )

        # Retrieve the posterior predictive probability for each design
        p_y = (
            torch.special.expit(pyro.param("q_logit"))
            .detach()
            .numpy()
        )
        p_outcome = dict()
        for idx, candidate in enumerate(candidates):
            p_outcome[candidate] = float(p_y[idx])

        # Find the item with maximum EIG
        best_idx = torch.argmax(eig)
        optimal_test = candidates[best_idx]
        best_eig = eig.detach().max()

        # Apply the early-stopping criterion
        epsilon = 0.04
        if best_eig < epsilon:
            print("Early stopping")
            return None

        if True:
            from matplotlib import pyplot as plt

            entropy = -p_y * np.log2(p_y) - (
                1 - p_y
            ) * np.log2(1 - p_y)

            x = np.linspace(-3, +3, 200)
            y = norm.pdf(
                x,
                loc=self.theta_means[
                    self.participant_index[participant]
                ],
                scale=self.theta_sds[
                    self.participant_index[participant]
                ],
            )

            plt.plot(
                x,
                y,
                color="black",
                label=r"$\theta$(participant)",
            )

            eig = eig.detach().numpy()

            cmap = plt.get_cmap("tab10")
            for i in range(len(candidates)):
                item = candidates[i]
                color = cmap(i % 10)
                y = norm.pdf(
                    x,
                    loc=self.difficulty_means[
                        self.item_index[item]
                    ]
                    - self.intercept_mean,
                    scale=np.sqrt(
                        self.difficulty_sds[
                            self.item_index[item]
                        ]
                        ** 2
                        + self.intercept_sd**2,
                    ),
                )
                plt.plot(
                    x,
                    y,
                    alpha=0.2,
                    color=color,
                )
                plt.scatter(
                    [
                        self.difficulty_means[
                            self.item_index[item]
                        ]
                        - self.intercept_mean,
                    ],
                    [eig[i]],
                    facecolors="none",
                    edgecolors=color,
                    marker="s",
                    label="EIG" if i == 0 else None,
                )

                plt.scatter(
                    [
                        self.difficulty_means[
                            self.item_index[item]
                        ]
                        - self.intercept_mean,
                    ],
                    [entropy[i]],
                    color=color,
                    label="$H(y)$" if i == 0 else None,
                )

            plt.axhline(epsilon, label=r"$\varepsilon$")
            plt.xlim(-3, 3)
            plt.ylim(0, 1)

            plt.legend()
            plt.savefig(
                "output/test_simul_{}.png".format(
                    participant
                ),
            )
            plt.clf()

            fig, ax = plt.subplots()
            x = np.linspace(0, 10, 200)
            for i in range(len(candidates)):
                item = candidates[i]
                y = lognorm.pdf(
                    x,
                    loc=self.log_tau_mus[
                        self.item_index[item]
                    ],
                    s=self.log_tau_sds[
                        self.item_index[item]
                    ],
                )
                color = cmap(i % 10)
                ax.set_xlim(0, 10)

                ax.plot(x, y, color=color)

            fig.savefig("output/test_simul_time_{}.png".format(participant))
            plt.clf()

        return optimal_test


class Simulator:
    def __init__(self, strategy, nodes, gamma=None):
        self.strategy = strategy
        self.nodes = nodes
        self.gamma = gamma
        self.data = {
            "participants": [],
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
        self.design = OptimalDesign()

    def add_participant(self, participant_id):
        self.data["participants"].append(
            int(participant_id)
        )

    def process_participant(self, participant_id):
        participant_trials = self.oracle_trials[
            self.oracle_trials["participant_id"]
            == participant_id
        ]

        if len(participant_trials) != 30:
            return

        skip_nodes = []
        n_treatments = 15
        for i in range(n_treatments):
            next_node = self.design.get_optimal_node(
                list(set(self.nodes) - set(skip_nodes)),
                int(participant_id),
                self.data,
                # self.strategy,
                # self.gamma,
            )

            if next_node is None:
                break

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
                "x": trial["time_taken"],
                "participant_id": int(
                    trial["participant_id"]
                ),
            }

            skip_nodes.append(next_node)

    def simulate(self):
        participants_ids = list(self.participants.keys())
        # random.shuffle(participants_ids)

        for participant_id in participants_ids:
            self.add_participant(participant_id)
            self.process_participant(participant_id)

        return self.data


# Example usage:
def simulate():

    # Run oracle simulation
    print("Running oracle simulation...")
    oracle = Simulator(
        strategy="active_inference",
        nodes=np.arange(15, 31),
        gamma=1,
    ).simulate()


if __name__ == "__main__":

    simulate()
