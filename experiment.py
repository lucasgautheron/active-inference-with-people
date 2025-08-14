# pylint: disable=unused-import,abstract-method

import logging

from markupsafe import Markup

import psynet.experiment
from psynet.modular_page import TextControl, ModularPage
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Timeline, CodeBlock
from psynet.participant import Participant
from psynet.trial.static import (
    StaticNode,
    StaticTrial,
    StaticTrialMaker,
)
from psynet.consent import MainConsent
from psynet.demography.general import (
    FormalEducation,
)

import torch
import pyro
import pyro.distributions as dist
import numpy as np
from torch.distributions.constraints import positive
from pyro.contrib.oed.eig import marginal_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from matplotlib import pyplot as plt
from scipy.stats import norm

import pandas as pd
import csv

DEBUG_MODE = True

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
)
logger = logging.getLogger()


class Oracle:
    """
    Oracle for simulating the experiment
    using real human data from Dubourg et al., 2025
    """

    def __init__(self, domain):
        answers = pd.read_csv("static/answers.csv")
        answers = np.stack(answers.values)[domain * 15 :, : (domain + 1) * 15]
        mask = ~np.any(pd.isna(answers), axis=1)
        answers = answers[mask]

        self.answers = [
            {
                "answers": answers[i],
            }
            for i in range(len(answers))
        ]

        self.education = pd.read_csv("static/education.csv")["college"].values[
            mask
        ]

        logger.info("Oracle data:")
        logger.info(answers.shape)

    def answer(self, participant_id: int, item: int):
        return self.answers[participant_id]["answers"][item]

    def college(self, participant_id: int):
        return self.education[participant_id]


oracle = Oracle(domain=0)


class AdaptiveTesting:
    def __init__(self):
        logger.debug("Initializing adaptive learner.")

        # Priors parameters
        self.prior_mean_theta = torch.tensor(0.0)
        self.prior_sd_theta = torch.tensor(2.0)
        self.prior_mean_difficulty = torch.tensor(0.0)
        self.prior_sd_difficulty = torch.tensor(1.0)
        self.prior_mean_intercept = torch.tensor(0.0)
        self.prior_sd_intercept = torch.tensor(1.0)

        self.theta_means = torch.empty(0)
        self.theta_sds = torch.empty(0)
        self.difficulty_means = torch.empty(0)
        self.difficulty_sds = torch.empty(0)
        self.intercept_mean = torch.tensor(0.0)
        self.intercept_sd = torch.tensor(0.0)

        # EIG computation parameters
        self.num_steps = 1000
        self.start_lr = 0.1
        self.end_lr = 0.001

    def _make_design_model(self, target_participant):
        """Create a model for a specific participant
        that takes item indices as design"""

        def model(design):
            with pyro.plate_stack("plate", design.shape[:-1]):
                # Sample ability parameter for the target participant
                theta = pyro.sample(
                    "theta",
                    dist.Normal(
                        self.theta_means[target_participant],
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
                logit_p = (theta - difficulties) + intercept

                y = pyro.sample(
                    "y",
                    dist.Bernoulli(
                        logits=logit_p,
                    ).to_event(1),
                )

                return y

        return model

    def _model(self, participants, items):
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

        selected_thetas = thetas[participants.long()]
        selected_difficulties = difficulties[items.long()]

        # Logistic regression model with intercept
        logit_p = (selected_thetas - selected_difficulties) + intercept
        y = pyro.sample(
            "y",
            dist.Bernoulli(
                logits=logit_p,
            ).to_event(1),
        )
        return y

    def _guide(self, participants, items):
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

        self.theta_means = torch.full([num_participants], self.prior_mean_theta)
        self.theta_sds = torch.full([num_participants], self.prior_sd_theta)
        self.difficulty_means = torch.full(
            [num_items], self.prior_mean_difficulty
        )
        self.difficulty_sds = torch.full([num_items], self.prior_sd_difficulty)
        self.intercept_mean = torch.tensor(self.prior_mean_intercept)
        self.intercept_sd = torch.tensor(self.prior_sd_intercept)

    def update_posterior(self, data):
        """Update posterior beliefs based on all collected experimental data"""

        participants = []
        items = []
        responses = []

        for node_id, trials in data["nodes"].items():
            for trial_id, trial_data in trials.items():
                participants.append(trial_data["participant_id"])
                items.append(node_id)
                responses.append(float(trial_data["y"]))

        self.participant_index = {
            participant: idx
            for idx, participant in enumerate(data["participants"])
        }
        self.item_index = {
            item: idx for idx, item in enumerate(data["nodes"].keys())
        }

        participants = torch.tensor(
            [
                self.participant_index[participant]
                for participant in participants
            ],
            dtype=torch.long,
        )
        items = torch.tensor(
            [self.item_index[item] for item in items], dtype=torch.long
        )
        responses = torch.tensor(responses)

        # Initialize parameters with correct sizes
        self.init_parameters(
            len(self.participant_index),
            len(self.item_index),
        )

        pyro.clear_param_store()

        # The statistical model conditioned on all prior responses
        conditioned_model = pyro.condition(self._model, {"y": responses})

        # Instantiate the stochastic variational inference
        svi = SVI(
            conditioned_model,
            self._guide,
            Adam({"lr": 0.02}),
            loss=Trace_ELBO(),
        )

        # Fit the model
        for i in range(self.num_steps):
            elbo = svi.step(participants, items)
            if i % 100 == 0:
                logger.debug(f"  Iteration {i}, ELBO: {elbo:.3f}")

        # Extract parameters
        self.theta_means = pyro.param("theta_means").detach().clone()
        self.theta_sds = pyro.param("theta_sds").detach().clone()
        self.difficulty_means = pyro.param("mean_difficulties").detach().clone()
        self.difficulty_sds = pyro.param("sd_difficulties").detach().clone()
        self.intercept_mean = pyro.param("mean_intercept").detach().clone()
        self.intercept_sd = pyro.param("sd_intercept").detach().clone()

        logger.debug("Posterior update completed")

    def get_optimal_node(self, candidates, participant, data):
        # Update posterior with current data
        self.update_posterior(data)

        # Create design model for this participant
        pyro.clear_param_store()
        design_model = self._make_design_model(
            self.participant_index[participant.id]
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
                "gamma": (self.end_lr / self.start_lr) ** (1 / self.num_steps),
            }
        )

        # Compute Expected Information Gain for each candidate item
        eig = marginal_eig(
            design_model,
            candidate_designs,
            "y",
            ["theta", "difficulties", "intercept"],
            num_samples=200 * 10,
            num_steps=self.num_steps * 2,
            guide=self._marginal_guide,
            optim=optimizer,
            final_num_samples=2000 * 10,
        )

        # Find the item with maximum EIG
        best_idx = torch.argmax(eig)
        optimal_test = candidates[best_idx]
        best_eig = eig.detach().max()

        # Apply early-stopping criteria
        p = torch.special.expit(pyro.param("q_logit")).detach().numpy()
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        epsilon = 0.05
        delta = 1
        if best_eig < epsilon and np.max(entropy) < delta:
            logger.debug("Stopping criteria met - low EIG and entropy")
            return None

        if DEBUG_MODE:
            x = np.linspace(-3, +3, 200)
            y = norm.pdf(
                x,
                loc=self.theta_means[self.participant_index[participant.id]],
                scale=self.theta_sds[self.participant_index[participant.id]],
            )

            plt.plot(x, y, color="black", label=r"$\theta$(participant)")

            eig = eig.detach().numpy()

            cmap = plt.get_cmap("tab10")
            for i in range(len(candidates)):
                item = candidates[i]
                color = cmap(i % 10)
                y = norm.pdf(
                    x,
                    loc=self.difficulty_means[self.item_index[item]]
                    - self.intercept_mean,
                    scale=np.sqrt(
                        self.difficulty_sds[self.item_index[item]] ** 2
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
                        self.difficulty_means[self.item_index[item]]
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
                        self.difficulty_means[self.item_index[item]]
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
                "output/test_{}.png".format(participant.id),
            )
            plt.clf()

        return optimal_test


class KnowledgeTrial(StaticTrial):
    time_estimate = (
        10  # how long it should take to complete each trial, in seconds
    )

    def __init__(
        self,
        experiment,
        node,
        participant,
        *args,
        **kwargs,
    ):
        """
        Initialize the trial
        """
        super().__init__(
            experiment,
            node,
            participant,
            *args,
            **kwargs,
        )

        # Keeps track of whether the participant correctly answered
        self.var.y = None

    def show_trial(self, experiment, participant):
        question = self.definition["question"]

        page = ModularPage(
            "knowledge_trial",
            Markup(
                f"""
                <p id='question'>{question}</p>
                (<i>Leave empty if you do not know</i>)
                """,
            ),
            TextControl(
                block_copy_paste=True,
                bot_response=lambda: oracle.answer(
                    participant.id,
                    self.definition["item_id"],
                ),
            ),
            time_estimate=self.time_estimate,
        )

        return page

    def show_feedback(self, experiment, participant):
        return InfoPage(
            (
                "Congratulations, this is correct!"
                if self.var.y == True
                else "Nice try, but no :("
            ),
        )


class KnowledgeTrialMaker(StaticTrialMaker):
    def __init__(
        self,
        optimizer_class,
        domain,
        z_attribute,
        *args,
        **kwargs,
    ):
        """
        Initialize the trial maker
        with the list of all possibles challenges
        """
        nodes = self.load_nodes(domain)

        super().__init__(
            *args,
            allow_repeated_nodes=False,
            # the class of the trials delivered
            trial_class=KnowledgeTrial,
            # do not repeat trials on the same participants,
            # (as is often done for assessing reliability)
            n_repeat_trials=0,
            # the list of all challenges
            nodes=nodes,
            **kwargs,
        )

        logger.info("Initializing optimization module.")
        self.optimizer = optimizer_class()
        self.z_attribute = z_attribute

    def load_nodes(self, domain):
        questions = pd.read_csv("static/questions.csv")
        questions["domain"] = questions["id"] // 15
        questions = questions[questions["domain"] == domain]
        logger.info(questions)

        nodes = [
            StaticNode(
                definition={
                    "item_id": i,
                    "question": question["question"],
                    "answers": question["answers"].split(
                        "|",
                    ),
                },
            )
            for i, question in enumerate(questions.to_dict(orient="records"))
        ]

        return nodes

    def prior_data(self, experiment):
        data = {"nodes": dict(), "participants": dict()}

        nodes = self.network_class.query.filter_by(
            trial_maker_id=self.id, full=False, failed=False
        )

        for node in nodes:
            data["nodes"][node.id] = dict()

            for trial in node.head.viable_trials:
                y = trial.var.get("y")
                z = (
                    trial.participant.var.get("z", None)
                    if self.z_attribute
                    else None
                )

                data["nodes"][node.id][trial.id] = {
                    "y": y,
                    "z": z,
                    "participant_id": trial.participant.id,
                }

        data["participants"] = {
            participant.id: {
                "z": (
                    participant.var.get("z", None) if self.z_attribute else None
                ),
            }
            for participant in Participant.query.all()
        }

        return data

    def prioritize_nodes(self, nodes, participant, experiment):
        candidates = {node.id: node for node in nodes}

        data = self.prior_data(experiment)
        next_node = self.optimizer.get_optimal_node(
            list(candidates.keys()), participant, data
        )

        if next_node is None:
            return []

        return [candidates[next_node]]

    def prioritize_networks(self, networks, participant, experiment):
        nodes = [network.head for network in networks]
        nodes = self.prioritize_nodes(nodes, participant, experiment)

        # filter
        networks = [
            network
            for network in networks
            if network.head.id in [node.id for node in nodes]
        ]

        # re-order
        order = [node.id for node in nodes]
        networks = sorted(
            networks, key=lambda network: order.index(network.head.id)
        )

        return networks

    def finalize_trial(
        self,
        answer,
        trial,
        experiment,
        participant,
    ):
        trial.var.y = trial.answer.lower() in trial.node.definition["answers"]

        super().finalize_trial(
            answer,
            trial,
            experiment,
            participant,
        )


class ActiveInference:
    def get_optimal_node(self, nodes_ids, participant, data):
        z_i = participant.var.z

        n_samples = 1000

        rewards = dict()
        eig = dict()
        utility = dict()

        alphas = dict()
        betas = dict()

        z_participants = np.array(
            [
                data["participants"][participant_id]["z"]
                for participant_id in data["participants"]
                if data["participants"][participant_id]["z"] != None
            ]
        )

        alpha_z = 1 + np.sum(z_participants == 1)
        beta_z = 1 + np.sum(z_participants == 0)
        p_z = alpha_z / (alpha_z + beta_z)

        for node_id in nodes_ids:
            alpha = np.ones(2)
            beta = np.ones(2)

            for trial_id, trial in data["nodes"][node_id].items():
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
                (2, n_samples),
            )

            y = np.random.binomial(
                np.ones((2, n_samples), dtype=int),
                phi,
                size=(
                    2,
                    n_samples,
                ),
            )

            p_y_given_phi = phi * y + (1 - phi) * (1 - y)
            p_y = alpha / (alpha + beta) * y + beta / (alpha + beta) * (1 - y)

            EIG = np.mean(np.log(p_y_given_phi[z_i] / p_y[z_i]))

            gamma = 0.2
            U = gamma * np.mean(
                p_z * y[1]
                + (1 - p_z) * (1 - y[0])
                - p_z * (1 - y[1])
                - (1 - p_z) * y[0]
            )

            rewards[node_id] = EIG + U
            eig[node_id] = EIG
            utility[node_id] = U

        from matplotlib import pyplot as plt

        if np.random.uniform() > 1 and len(nodes_ids) == 15:
            cmap = plt.get_cmap("tab10")
            fig, ax = plt.subplots()
            for node_id in nodes_ids:
                color = cmap(node_id % 10)
                x = np.linspace(0, 1, 100)
                ax.plot(
                    x,
                    beta_dist.pdf(
                        x,
                        a=alphas[node_id][0],
                        b=betas[node_id][0],
                    ),
                    color=color,
                    label=rewards[node_id],
                )
                ax.plot(
                    x,
                    beta_dist.pdf(
                        x,
                        a=alphas[node_id][1],
                        b=betas[node_id][1],
                    ),
                    color=color,
                    ls="dashed",
                )
            plt.legend()
            plt.show()

        best_network = sorted(
            list(rewards.keys()),
            key=lambda node_id: rewards[node_id],
            reverse=True,
        )[0]

        if len(nodes_ids) == 15:
            logger.info("Saving")
            with open("output/utility.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        rewards[best_network],
                        eig[best_network],
                        utility[best_network],
                    ]
                )

        return best_network


class Exp(psynet.experiment.Experiment):
    label = "Active inference for adaptive experiments"
    initial_recruitment_size = 1
    test_n_bots = 200
    test_mode = "serial"

    timeline = Timeline(
        MainConsent(),
        FormalEducation(),
        CodeBlock(
            lambda participant: participant.var.set(
                "z",
                (
                    oracle.college(participant.id)
                    if DEBUG_MODE
                    else (
                        participant.answer
                        in [
                            "college",
                            "graduate_school",
                            "postgraduate_degree_or_higher",
                        ]
                    )
                ),
            )
        ),
        KnowledgeTrialMaker(
            id_="optimal_treatment",
            optimizer_class=ActiveInference,
            domain=0,
            z_attribute=True,
            expected_trials_per_participant=5,
            max_trials_per_participant=5,
        ),
        KnowledgeTrialMaker(
            id_="optimal_test",
            optimizer_class=AdaptiveTesting,
            domain=0,
            z_attribute=False,
            expected_trials_per_participant=15,
            max_trials_per_participant=15,
        ),
        SuccessfulEndPage(),
    )
