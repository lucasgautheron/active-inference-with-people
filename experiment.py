# pylint: disable=unused-import,abstract-method

import logging

from markupsafe import Markup

import psynet.experiment
from psynet.modular_page import TextControl, ModularPage
from psynet.page import InfoPage
from psynet.timeline import Timeline
from psynet.trial.static import (
    StaticNetwork, StaticNode, StaticTrial,
    StaticTrialMaker,
)
from psynet.consent import MainConsent
from psynet.demography.general import BasicDemography, Income

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
from os.path import exists

import pandas as pd

DEBUG_MODE = True

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger()


class Oracle:
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


class AdaptiveLearner:
    """Adaptive Bayesian Learner"""

    def __init__(self, num_items):
        logger.debug("Initializing adaptive learner.")

        # Number of challenges
        self.num_items = num_items

        # Initialize data storage
        self.participant_indices = torch.empty(0, dtype=torch.long)
        self.item_indices = torch.empty(0, dtype=torch.long)
        self.responses = torch.empty(0)

        self.participants = {}

        # Priors parameters
        self.prior_mean_theta = torch.tensor(0.0)
        self.prior_sd_theta = torch.tensor(2.0)
        self.prior_mean_difficulty = torch.tensor(0.0)
        self.prior_sd_difficulty = torch.tensor(1.0)
        self.prior_mean_intercept = torch.tensor(0.0)
        self.prior_sd_intercept = torch.tensor(1.0)

        # Current posterior estimates for individual knowledge
        self.posterior_theta_means = torch.empty(0)
        self.posterior_theta_sds = torch.empty(0)

        # Current posterior estimates for difficulties
        self.posterior_difficulty_means = torch.tensor(
            [0.0] * self.num_items,
        )
        self.posterior_difficulty_sds = torch.tensor(
            [1.0] * self.num_items,
        )

        # Current posterior estimates for intercept
        self.posterior_intercept_mean = torch.tensor(self.prior_mean_intercept)
        self.posterior_intercept_sd = torch.tensor(self.prior_sd_intercept)

        # EIG computation parameters
        self.num_steps = 1000
        self.start_lr = 0.1
        self.end_lr = 0.001

    def add_participant(self, participant):
        """Add a new participant to the study"""
        new_participant_id = len(self.participants)
        self.participants[new_participant_id] = participant.id

        # Expand theta tracking tensors for new participant
        self.posterior_theta_means = torch.cat(
            [
                self.posterior_theta_means,
                torch.tensor([self.prior_mean_theta]),
                # Prior mean for new participant
            ],
        )
        self.posterior_theta_sds = torch.cat(
            [
                self.posterior_theta_sds,
                torch.tensor([self.prior_sd_theta]),
                # Prior std for new participant
            ],
        )

        logger.debug(
            f"AdaptiveLearner added new participant: {new_participant_id}",
        )

        return new_participant_id

    def get_participant(self, participant):
        for pid, participant_id in self.participants.items():
            if participant.id == participant_id:
                return pid

        raise Exception(f"Participant {participant.id} not found")

        def _make_design_model(self, target_participant):
            """Create a model for a specific participant that takes item indices as design"""

            def model(design):
                # Sample ability parameter for the target participant
                theta = pyro.sample(
                    "theta",
                    dist.Normal(
                        self.posterior_theta_means[target_participant],
                        self.posterior_theta_sds[target_participant],
                    ),
                )
                theta = theta.unsqueeze(-1)

                item_idx = design.squeeze(-1).long()
                difficulties = pyro.sample(
                    "difficulties",
                    dist.Normal(
                        self.posterior_difficulty_means[item_idx],
                        self.posterior_difficulty_sds[item_idx],
                    ),
                ).unsqueeze(-1)

                intercept = pyro.sample(
                    "intercept", dist.Normal(
                        self.posterior_intercept_mean,
                        self.posterior_intercept_sd,
                    ),
                ).unsqueeze(-1)
                logit_p = (theta - difficulties) + intercept

                y = pyro.sample("y", dist.Bernoulli(logits=logit_p).to_event(1))

                return y

            return model

    def _model(self, participant_indices, item_indices):
        """Model of the data-generating process"""
        # Get the maximum participant ID to determine how many participants we have
        max_participant = int(participant_indices.max().item()) + 1

        # Sample ability parameters for all participants: theta_i ~ N(0, 2)
        thetas = pyro.sample(
            "thetas",
            dist.Normal(self.prior_mean_theta, self.prior_sd_theta).expand(
                [max_participant],
            ).to_event(1),
        )

        # Sample difficulty parameters for all potential items
        difficulties = pyro.sample(
            "difficulties",
            dist.Normal(
                self.prior_mean_difficulty,
                self.prior_sd_difficulty,
            ).expand(
                [self.num_items],
            ).to_event(1),
        )

        # Sample intercept parameter
        intercept = pyro.sample(
            "intercept",
            dist.Normal(self.prior_mean_intercept, self.prior_sd_intercept),
        )

        # Select abilities and difficulties for the observations
        selected_thetas = thetas[participant_indices.long()]
        selected_difficulties = difficulties[item_indices.long()]

        # Logistic regression model with intercept
        logit_p = (selected_thetas - selected_difficulties) + intercept
        y = pyro.sample("y", dist.Bernoulli(logits=logit_p).to_event(1))
        return y

    def _guide(self, participant_indices, item_indices):
        """Guide for multiple participants with hierarchical theta structure"""
        # Get the maximum participant ID to determine how many participants we have
        max_participant = int(participant_indices.max().item()) + 1

        # Guide for thetas - individual posterior means and standard deviations
        posterior_theta_means = pyro.param(
            "posterior_theta_means",
            torch.full([max_participant], self.prior_mean_theta),
        )
        posterior_theta_sds = pyro.param(
            "posterior_theta_sds",
            torch.full([max_participant], self.prior_sd_theta),
            constraint=positive,
        )
        pyro.sample(
            "thetas", dist.Normal(
                posterior_theta_means,
                posterior_theta_sds,
            ).to_event(1),
        )

        # Guide for difficulties - for all potential items
        posterior_mean_difficulties = pyro.param(
            "posterior_mean_difficulties",
            self.prior_mean_difficulty.expand([self.num_items]).clone(),
        )
        posterior_sd_difficulties = pyro.param(
            "posterior_sd_difficulties",
            self.prior_sd_difficulty.expand([self.num_items]).clone(),
            constraint=positive,
        )
        pyro.sample(
            "difficulties",
            dist.Normal(
                posterior_mean_difficulties,
                posterior_sd_difficulties,
            ).to_event(1),
        )

        # Guide for intercept
        posterior_mean_intercept = pyro.param(
            "posterior_mean_intercept",
            self.prior_mean_intercept.clone(),
        )
        posterior_sd_intercept = pyro.param(
            "posterior_sd_intercept",
            self.prior_sd_intercept.clone(),
            constraint=positive,
        )
        pyro.sample(
            "intercept",
            dist.Normal(posterior_mean_intercept, posterior_sd_intercept),
        )

    def _marginal_guide(self, design, observation_labels, target_labels):
        """Guide for marginal_eig - design is a single item index"""
        q_logit = pyro.param("q_logit", torch.zeros(design.shape[-2:]))
        pyro.sample("y", dist.Bernoulli(logits=q_logit).to_event(1))

    def get_optimal_test(self, participant_id: int, candidates: np.array):
        """Find the optimal test item for a given participant"""
        # Create design model for this participant
        pyro.clear_param_store()
        design_model = self._make_design_model(participant_id)

        # Candidate designs (available items)
        candidate_designs = torch.tensor(
            candidates,
            dtype=torch.float,
        ).unsqueeze(-1)

        if len(candidates) == 0:
            raise ValueError("No available items to choose from")

        optimizer = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": self.start_lr},
                "gamma": (self.end_lr / self.start_lr) ** (1 / self.num_steps),
            },
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

        p = torch.special.expit(pyro.param("q_logit")).detach().numpy()
        entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

        # Find the item with maximum EIG
        best_idx = torch.argmax(eig)
        optimal_item = candidates[best_idx]

        if DEBUG_MODE:
            x = np.linspace(-3, +3, 200)
            y = norm.pdf(
                x, loc=self.posterior_theta_means[participant_id],
                scale=self.posterior_theta_sds[participant_id],
            )

            plt.plot(x, y, label="Participant ability")

            cmap = plt.get_cmap("tab10")
            for i in range(len(candidates)):
                item = candidates[i]
                color = cmap(i)
                y = norm.pdf(
                    x, loc=self.posterior_difficulty_means[
                               item] - self.posterior_intercept_mean,
                    scale=np.sqrt(
                        self.posterior_difficulty_sds[
                            item] ** 2 + self.posterior_intercept_sd ** 2,
                    ),
                )
                plt.plot(x, y, label="Item difficulty", alpha=0.2, color=color)
                plt.scatter(
                    [
                        self.posterior_difficulty_means[
                            item] - self.posterior_intercept_mean,
                    ],
                    [eig.detach()[i]],
                    color=color,
                )

                plt.scatter(
                    [
                        self.posterior_difficulty_means[
                            item] - self.posterior_intercept_mean,
                    ],
                    [entropy[i]],
                    color="black",
                )
                plt.xlim(-3, 3)
                plt.ylim(0, 1)

            plt.savefig("output/test_{}.png".format(participant_id))
            plt.clf()

        return optimal_item, eig.detach().max(), eig, entropy

    def administer_response(self, participant_id, item_id, response):
        """Record a participant's response to an item"""
        # Convert response to tensor if it's not already
        if not isinstance(response, torch.Tensor):
            response = torch.tensor(float(response))

        # Store the result
        self.participant_indices = torch.cat(
            [self.participant_indices, torch.tensor([participant_id])], dim=0,
        )
        self.item_indices = torch.cat(
            [self.item_indices, torch.tensor([item_id])], dim=0,
        )
        self.responses = torch.cat([self.responses, response.expand(1)])

    def update_posterior(self):
        """Update posterior beliefs based on collected data"""
        if len(self.responses) == 0:
            return

        logger.debug("Updating posterior...")
        pyro.clear_param_store()

        # The statistical model conditioned on all prior responses :
        conditioned_model = pyro.condition(self._model, {"y": self.responses})

        # Instantiate the stochastic variational inference
        svi = SVI(
            conditioned_model,  # True model
            self._guide,  # The variational approximation of the posterior
            Adam({"lr": 0.02}),  # The optimization algorithm
            loss=Trace_ELBO(),
        )

        # Fit the model
        for i in range(self.num_steps):
            elbo = svi.step(self.participant_indices, self.item_indices)
            if i % 100 == 0:
                logger.debug(f"  Iteration {i}, ELBO: {elbo:.3f}")

        # Extract updated parameters
        self.posterior_theta_means = pyro.param(
            "posterior_theta_means",
        ).detach().clone()
        self.posterior_theta_sds = pyro.param(
            "posterior_theta_sds",
        ).detach().clone()

        self.posterior_difficulty_means = pyro.param(
            "posterior_mean_difficulties",
        ).detach().clone()
        self.posterior_difficulty_sds = pyro.param(
            "posterior_sd_difficulties",
        ).detach().clone()
        self.posterior_intercept_mean = pyro.param(
            "posterior_mean_intercept",
        ).detach().clone()
        self.posterior_intercept_sd = pyro.param(
            "posterior_sd_intercept",
        ).detach().clone()

        np.savez(
            'posterior_parameters.npz',
            theta_means=self.posterior_theta_means.numpy(),
            theta_sds=self.posterior_theta_sds.numpy(),
            difficulty_means=self.posterior_difficulty_means.numpy(),
            difficulty_sds=self.posterior_difficulty_sds.numpy(),
            intercept_mean=self.posterior_intercept_mean.numpy(),
            intercept_sd=self.posterior_intercept_sd.numpy(),
            participant_indices=self.participant_indices.numpy(),
            item_indices=self.item_indices.numpy(),
            responses=self.responses.numpy(),
        )


class KnowledgeTrial(StaticTrial):
    time_estimate = 10  # how long it should take to complete each trial, in seconds

    def __init__(self, experiment, node, participant, *args, **kwargs):
        """
        Initialize the trial
        """
        super().__init__(experiment, node, participant, *args, **kwargs)

        # Keeps track of whether the participant correctly answered
        self.var.correct = None
        self.var.ability_mean = None
        self.var.ability_sd = None

    def show_trial(self, experiment, participant):
        question = self.definition["question"]

        page = ModularPage(
            "knowledge_trial",
            Markup(
                f"""
                <p id='question'>{question}</p>
                (Leave empty if you do not know)
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
            Markup("Congratulations, this is correct!")
            if self.var.correct == True
            else Markup("Nice try, but no :("),
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
            # how many questions administered to each participant (estimate)
            expected_trials_per_participant=15,
            # how many questions administered to each participant (maximum)
            max_trials_per_participant=15,
            # can the same question be shown multiple times?
            allow_repeated_nodes=False,
            # the class of the trials delivered by the trial maker
            trial_class=KnowledgeTrial,
            # do not repeat trials on participants,
            # as is often done for assessing reliability
            n_repeat_trials=0,
            # the list of all challenges
            nodes=nodes,
        )

        logger.info("Initializing adaptive learner.")
        self.learner = AdaptiveLearner(len(nodes))

    def load_nodes(self):
        questions = pd.read_csv("static/questions.csv")
        questions["domain"] = questions["id"] // 15
        questions = questions[questions["domain"] == 0]

        nodes = [
            StaticNode(
                definition={
                    "item_id": i,
                    "question": question["question"],
                    "answers": question["answers"].split("|"),
                },
            )
            for i, question in enumerate(questions.to_dict(orient="records"))
        ]

        return nodes

    def custom_network_filter(self, candidates, participant):
        logger.info("Getting data for participant")

        if participant.id not in self.learner.participants.values():
            logger.info("New participant")
            pid = self.learner.add_participant(participant)
            logger.info(self.learner.participants)
        else:
            pid = self.learner.get_participant(participant)

        candidate_items = []

        logger.info(candidates)
        for candidate in candidates:
            candidate_items += [node.definition["item_id"] for node in
                                candidate.nodes()]

        # choose best item
        next_node_id, eig_value, eig_scores, entropies = self.learner.get_optimal_test(
            pid,
            np.array(candidate_items),
        )

        if eig_value < 0.02 and np.max(entropies) < 0.7:
            return []

        # recover the retained candidate
        for candidate in candidates:
            if any(
                    [node.definition["item_id"] == next_node_id for node in
                     candidate.nodes()],
            ):
                return [candidate]

        return candidates

    def finalize_trial(self, answer, trial, experiment, participant):
        trial.var.correct = trial.answer.lower() in trial.node.definition[
            "answers"]

        logger.info("Finalizing trial, looking for participant")
        if participant.id not in self.learner.participants.values():
            logger.info("New participant")
            pid = self.learner.add_participant(participant)
        else:
            pid = self.learner.get_participant(participant)

        self.learner.administer_response(
            pid, trial.node.definition["item_id"],
            trial.var.correct,
        )

        self.learner.update_posterior()

        mu = self.learner.posterior_theta_means[pid]
        sd = self.learner.posterior_theta_sds[pid]
        logger.info(f"Posterior participant ability: N({mu:.2f}, {sd:.2f})")
        trial.var.ability_mean = float(mu)
        trial.var.ability_sd = float(sd)

        mu = self.learner.posterior_difficulty_means[
            trial.node.definition["item_id"]]
        sd = self.learner.posterior_difficulty_sds[
            trial.node.definition["item_id"]]
        logger.info(f"Posterior item difficulty: N({mu:.2f}, {sd:.2f})")
        logger.info(trial.node.definition["question"])
        logger.info(trial.answer)

        super().finalize_trial(answer, trial, experiment, participant)


class Exp(psynet.experiment.Experiment):
    label = "Adaptive Bayesian testing demo"
    initial_recruitment_size = 1
    test_n_bots = 200
    test_mode = "serial"

    timeline = Timeline(
        MainConsent(),
        BasicDemography(),
        Income(),
        KnowledgeTrialMaker(),
    )
