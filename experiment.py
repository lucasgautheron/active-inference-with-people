# pylint: disable=unused-import,abstract-method

import logging

from markupsafe import Markup

import psynet.experiment
from psynet.modular_page import TextControl, ModularPage
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Timeline
from psynet.participant import Participant
from psynet.trial.static import (
    StaticNode,
    StaticTrial,
    StaticTrialMaker,
)
from psynet.consent import MainConsent
from psynet.demography.general import (
    BasicDemography,
    Income,
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

    def __init__(self):
        answers = pd.read_csv("static/answers.csv")
        answers = np.stack(answers.values)[:, :15]
        answers = answers[~np.any(pd.isna(answers), axis=1)]

        self.answers = [
            {
                "answers": answers[i],
            }
            for i in range(len(answers))
        ]

    def answer(self, participant_id: int, item: int):
        return self.answers[participant_id]["answers"][item]


oracle = Oracle()


class ActiveInference:
    """Adaptive Bayesian Learner"""

    def __init__(self, num_items):
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
        # Get the maximum participant ID
        # to determine how many participants we have

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

        # Select abilities and difficulties
        # for the observations
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

        for network_id, trials in data["networks"].items():
            for trial_id, trial_data in trials.items():
                y = trial_data["y"]
                participants.append(trial_data["participant_id"])
                items.append(network_id)
                responses.append(float(y))

            logger.info(responses)

        participants = torch.tensor(participants, dtype=torch.long)
        items = torch.tensor(items, dtype=torch.long)
        responses = torch.tensor(responses)

        # Initialize parameters with correct sizes
        self.init_parameters(
            np.max(data["participants"]),
            len(data["networks"]),
        )

        logger.info("theta")
        logger.info(self.theta_means.shape)

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
            elbo = svi.step(participants - 1, items - 1)
            if i % 100 == 0:
                logger.debug(f"  Iteration {i}, ELBO: {elbo:.3f}")

        # Extract updated parameters
        self.theta_means = pyro.param("theta_means").detach().clone()
        self.theta_sds = pyro.param("theta_sds").detach().clone()
        self.difficulty_means = pyro.param("mean_difficulties").detach().clone()
        self.difficulty_sds = pyro.param("sd_difficulties").detach().clone()
        self.intercept_mean = pyro.param("mean_intercept").detach().clone()
        self.intercept_sd = pyro.param("sd_intercept").detach().clone()

        logger.debug("Posterior update completed")

    def get_optimal_test(self, candidates, participant, data):
        # Update posterior with current data
        self.update_posterior(data)

        # Get available items for this participant
        # This is a simplified version - you'll need to map networks to items properly
        available_items = np.array(candidates)

        # Create design model for this participant
        pyro.clear_param_store()
        design_model = self._make_design_model(participant.id - 1)

        # Candidate designs (available items)
        candidate_designs = torch.tensor(
            available_items - 1, dtype=torch.float
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
        optimal_test = available_items[best_idx]
        best_eig = eig.detach().max()

        # Apply stopping criteria
        p = torch.special.expit(pyro.param("q_logit")).detach().numpy()
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        if best_eig < 0.02 and np.max(entropy) < 0.7:
            logger.debug("Stopping criteria met - low EIG and entropy")
            return None

        if DEBUG_MODE:
            x = np.linspace(-3, +3, 200)
            y = norm.pdf(
                x,
                loc=self.theta_means[participant.id - 1],
                scale=self.theta_sds[participant.id - 1],
            )

            plt.plot(x, y, label="Participant ability")

            cmap = plt.get_cmap("tab10")
            for i in range(len(candidates)):
                item = candidates[i]
                color = cmap(i)
                y = norm.pdf(
                    x,
                    loc=self.difficulty_means[item - 1] - self.intercept_mean,
                    scale=np.sqrt(
                        self.difficulty_sds[item - 1] ** 2
                        + self.intercept_sd**2,
                    ),
                )
                plt.plot(
                    x,
                    y,
                    label="Item difficulty",
                    alpha=0.2,
                    color=color,
                )
                plt.scatter(
                    [
                        self.difficulty_means[item - 1] - self.intercept_mean,
                    ],
                    [eig.detach()[i]],
                    color=color,
                )

                plt.scatter(
                    [
                        self.difficulty_means[item - 1] - self.intercept_mean,
                    ],
                    [entropy[i]],
                    color="black",
                )
                plt.xlim(-3, 3)
                plt.ylim(0, 1)

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
        self.var.ability_mean = None
        self.var.ability_sd = None

    def get_y(self):
        return self.var.y

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
    def __init__(self, *args, **kwargs):
        """
        Initialize the trial maker
        with the list of all possibles challenges
        """
        nodes = self.load_nodes()

        super().__init__(
            id_="knowledge",
            # number questions administered
            # to each participant (estimate)
            expected_trials_per_participant=15,
            # number of questions administered
            # to each participant (maximum)
            max_trials_per_participant=15,
            # can the same question be shown multiple times?
            allow_repeated_nodes=False,
            # the class of the trials delivered
            trial_class=KnowledgeTrial,
            # do not repeat trials on the same participants,
            # (as is often done for assessing reliability)
            n_repeat_trials=0,
            # the list of all challenges
            nodes=nodes,
        )

        logger.info("Initializing adaptive learner.")
        self.ai = ActiveInference(len(nodes))

    def load_nodes(self):
        questions = pd.read_csv("static/questions.csv")
        questions["domain"] = questions["id"] // 15
        questions = questions[questions["domain"] == 0]

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
        data = {"networks": dict(), "participants": []}

        networks = self.network_class.query.filter_by(
            trial_maker_id=self.id, full=False, failed=False
        )

        for network in networks:
            data["networks"][network.id] = dict()

            for trial in network.head.viable_trials:
                y = trial.get_y()

                data["networks"][network.id][trial.id] = {
                    "y": y,
                    "participant_id": trial.participant.id,
                }

        data["participants"] = [
            participant.id for participant in Participant.query.all()
        ]

        return data

    def prioritize_networks(self, networks, participant, experiment):
        candidates = {network.id: network for network in networks}

        data = self.prior_data(experiment)
        next_network = self.ai.get_optimal_test(
            list(candidates.keys()), participant, data
        )

        if next_network is None:
            return []

        return [candidates[next_network]]

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


class Exp(psynet.experiment.Experiment):
    label = "Adaptive Bayesian testing demo"
    initial_recruitment_size = 1
    test_n_bots = 200
    test_mode = "serial"

    timeline = Timeline(
        # MainConsent(),
        # BasicDemography(),
        # Income(),
        KnowledgeTrialMaker(),
        SuccessfulEndPage(),
    )
