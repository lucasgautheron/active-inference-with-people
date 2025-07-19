# pylint: disable=unused-import,abstract-method

import logging

from markupsafe import Markup

import psynet.experiment
from psynet.modular_page import TextControl, ModularPage
from psynet.page import InfoPage
from psynet.timeline import Timeline
from psynet.trial.static import StaticNetwork, StaticNode, StaticTrial, StaticTrialMaker

import torch
import pyro
import pyro.distributions as dist
import numpy as np
from torch.distributions.constraints import positive
from pyro.contrib.oed.eig import marginal_eig, posterior_eig, vnmc_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam

from matplotlib import pyplot as plt
from scipy.stats import norm

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Oracle:
    def __init__(self):
        answers = pd.read_csv("static/answers.csv")
        answers = np.stack(answers.values)[:, :15]
        answers = answers[~np.any(pd.isna(answers), axis=1)]

        self.answers = [
            {
                "answers": answers[i]
            }
            for i in range(len(answers))
        ]

    def answer(self, participant_id: int, item: int):
        return self.answers[participant_id]["answers"][item]


oracle = Oracle()


class AdaptiveLearner:
    """Adaptive test design system using Bayesian optimization"""

    def __init__(self, num_items, max_participants=1000):
        logger.debug("Initializing adaptive learner.")

        self.max_participants = max_participants
        self.num_items = num_items

        # Prior parameters
        self.prior_mean_theta = torch.tensor(0.0)
        self.prior_sd_theta = torch.tensor(1.0)
        self.prior_mean_difficulty = torch.tensor(0.0)
        self.prior_sd_difficulty = torch.tensor(1.0)

        # Prior parameters for intercept term
        self.prior_mean_intercept = torch.tensor(0.0)
        self.prior_sd_intercept = torch.tensor(1.0)

        # Initialize data storage
        self.participant_indices = torch.empty(0, dtype=torch.long)
        self.item_indices = torch.empty(0, dtype=torch.long)
        self.responses = torch.empty(0)

        self.participants = {}
        self.current_participants = []

        self.current_theta_means = torch.tensor(
            [self.prior_mean_theta] * max_participants
        )
        self.current_theta_sds = torch.tensor([self.prior_sd_theta] * max_participants)
        self.current_difficulty_means = torch.tensor(
            [self.prior_mean_difficulty] * self.num_items
        )
        self.current_difficulty_sds = torch.tensor(
            [self.prior_sd_difficulty] * self.num_items
        )

        self.current_intercept_mean = torch.tensor(self.prior_mean_intercept)
        self.current_intercept_sd = torch.tensor(self.prior_sd_intercept)

        # EIG computation parameters
        self.num_steps = 2000
        self.start_lr = 0.1
        self.end_lr = 0.001

    def add_participant(self, participant):
        """Add a new participant to the study"""
        new_participant_id = len(self.current_participants)
        self.participants[new_participant_id] = participant.id
        self.current_participants.append(new_participant_id)
        logger.debug(f"AdaptiveLearner added new participant: {new_participant_id}")

        return new_participant_id

    def get_participant(self, participant):
        for pid, participant_id in self.participants.items():
            if participant.id == participant_id:
                return pid

        raise Exception(f"Participant {participant.id} not found")

    def _make_design_model(self, target_participant):
        """Create a model for a specific participant that takes item indices as design"""

        def model(design):
            # Design has shape (1,) containing a single item index
            item_idx = design.squeeze().long()

            # Sample ability parameter for the target participant
            theta = pyro.sample(
                "theta",
                dist.Normal(
                    self.current_theta_means[target_participant],
                    self.current_theta_sds[target_participant],
                ),
            )

            # Sample difficulty parameters for all potential items
            difficulties = pyro.sample(
                "difficulties",
                dist.Normal(self.current_difficulty_means, self.current_difficulty_sds),
            )

            # Sample intercept parameter
            intercept = pyro.sample(
                "intercept",
                dist.Normal(self.current_intercept_mean, self.current_intercept_sd),
            )

            # Select difficulty for the item being used
            selected_difficulty = difficulties[item_idx]

            # Logistic regression model with intercept
            logit_p = (theta - selected_difficulty) + intercept
            y = pyro.sample("y", dist.Bernoulli(logits=logit_p))
            return y

        return model

    def _model(self, participant_indices, item_indices):
        """General model that handles multiple participants and items"""
        # Get the maximum participant ID to determine how many participants we have
        max_participant = int(participant_indices.max().item()) + 1

        # Sample ability parameters for all participants seen so far
        thetas = pyro.sample(
            "thetas",
            dist.Normal(self.prior_mean_theta, self.prior_sd_theta).expand(
                [max_participant]
            ).to_event(1),
        )

        # Sample difficulty parameters for all potential items
        difficulties = pyro.sample(
            "difficulties",
            dist.Normal(self.prior_mean_difficulty, self.prior_sd_difficulty).expand(
                [self.num_items]
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
        """Guide for multiple participants"""
        # Get the maximum participant ID to determine how many participants we have
        max_participant = int(participant_indices.max().item()) + 1

        # Guide for thetas - for all participants seen so far
        posterior_mean_thetas = pyro.param(
            "posterior_mean_thetas",
            self.prior_mean_theta.expand([max_participant]).clone(),
        )
        posterior_sd_thetas = pyro.param(
            "posterior_sd_thetas",
            self.prior_sd_theta.expand([max_participant]).clone(),
            constraint=positive,
        )
        pyro.sample("thetas", dist.Normal(posterior_mean_thetas, posterior_sd_thetas).to_event(1))

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
            dist.Normal(posterior_mean_difficulties, posterior_sd_difficulties).to_event(1),
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
            "intercept", dist.Normal(posterior_mean_intercept, posterior_sd_intercept)
        )

    def _marginal_guide(self, design, observation_labels, target_labels):
        """Guide for marginal_eig - design is a single item index"""
        q_logit = pyro.param("q_logit", torch.zeros(1))
        pyro.sample("y", dist.Bernoulli(logits=q_logit))

    def get_optimal_test(self, participant_id, exclude: list = []):
        """Find the optimal test item for a given participant"""
        # Create design model for this participant
        pyro.clear_param_store()
        design_model = self._make_design_model(participant_id)

        # Candidate designs for this participant (all possible items)
        candidate_designs = torch.arange(
            self.num_items, dtype=torch.float
        ).unsqueeze(-1)

        optimizer = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": self.start_lr},
                "gamma": (self.end_lr / self.start_lr) ** (1 / self.num_steps),
            }
        )

        eig = marginal_eig(
            design_model,
            candidate_designs,
            "y",
            [
                "theta",
                "difficulties",
                "intercept",
            ],  # Added intercept to target labels
            num_samples=100,
            num_steps=self.num_steps,
            guide=self._marginal_guide,
            optim=optimizer,
            final_num_samples=10000,
        )

        x = np.linspace(-3, +3, 200)
        y = norm.pdf(x, loc=self.current_theta_means[participant_id], scale=self.current_theta_sds[participant_id])
        plt.plot(x, y, label="Participant ability")

        cmap = plt.get_cmap("tab10")
        for i in range(self.num_items):
            color = cmap(i)
            y = norm.pdf(x, loc=self.current_difficulty_means[i], scale=self.current_difficulty_sds[i])
            plt.plot(x, y, label="Participant ability", alpha=0.2, color=color)
            plt.scatter([self.current_difficulty_means[i]], [eig.detach()[i]], color=color)

        plt.savefig("test_{}.png".format(participant_id))
        plt.clf()

        # Find best item for this participant
        mask = torch.ones_like(eig, dtype=torch.bool)
        if exclude:
            exclude_tensor = torch.tensor(exclude, device=eig.device)
            mask[exclude_tensor] = False

        # Set excluded items to negative infinity so they won't be selected
        masked_eig = eig.clone()
        masked_eig[~mask] = float("-inf")

        # Find the best item from the remaining candidates
        best_item = torch.argmax(masked_eig).long()
        best_eig = eig[best_item]

        return best_item.item(), best_eig.item(), eig

    def get_optimal_test_for_participant(self, participant_id, exclude: list = []):
        """Get the optimal test item for a given participant"""
        item_id, eig_value, eig_scores = self.get_optimal_test(participant_id, exclude)
        return item_id

    def administer_response(self, participant_id, item_id, response):
        """Record a participant's response to an item"""
        # Convert response to tensor if it's not already
        if not isinstance(response, torch.Tensor):
            response = torch.tensor(float(response))

        # Store the result
        self.participant_indices = torch.cat(
            [self.participant_indices, torch.tensor([participant_id])], dim=0
        )
        self.item_indices = torch.cat(
            [self.item_indices, torch.tensor([item_id])], dim=0
        )
        self.responses = torch.cat([self.responses, response.expand(1)])

    def update_posterior(self):
        """Update posterior beliefs based on collected data"""
        if len(self.responses) == 0:
            return

        logger.debug("Updating posterior...")
        pyro.clear_param_store()
        conditioned_model = pyro.condition(self._model, {"y": self.responses})
        svi = SVI(
            conditioned_model,
            self._guide,
            Adam({"lr": 0.02}),
            loss=Trace_ELBO(),
        )

        # Fit the model
        for i in range(self.num_steps):
            elbo = svi.step(self.participant_indices, self.item_indices)
            if i % 100 == 0:
                logger.debug(f"  Iteration {i}, ELBO: {elbo:.3f}")

        # Extract updated parameters
        posterior_theta_means = pyro.param("posterior_mean_thetas").detach()
        posterior_theta_sds = pyro.param("posterior_sd_thetas").detach()
        posterior_difficulty_means = pyro.param("posterior_mean_difficulties").detach()
        posterior_difficulty_sds = pyro.param("posterior_sd_difficulties").detach()
        posterior_intercept_mean = pyro.param("posterior_mean_intercept").detach()
        posterior_intercept_sd = pyro.param("posterior_sd_intercept").detach()

        # Resize tracking tensors if needed
        if posterior_theta_means.size(0) > self.current_theta_means.size(0):
            new_size = posterior_theta_means.size(0)
            new_theta_means = torch.full((new_size,), self.prior_mean_theta)
            new_theta_sds = torch.full((new_size,), self.prior_sd_theta)

            # Copy existing values
            new_theta_means[: self.current_theta_means.size(0)] = (
                self.current_theta_means[: self.current_theta_means.size(0)]
            )
            new_theta_sds[: self.current_theta_sds.size(0)] = self.current_theta_sds[
                                                              : self.current_theta_sds.size(0)
                                                              ]

            self.current_theta_means = new_theta_means
            self.current_theta_sds = new_theta_sds

        # Update with posterior values
        self.current_theta_means[: posterior_theta_means.size(0)] = (
            posterior_theta_means
        )
        self.current_theta_sds[: posterior_theta_sds.size(0)] = posterior_theta_sds
        self.current_difficulty_means = posterior_difficulty_means
        self.current_difficulty_sds = posterior_difficulty_sds
        self.current_intercept_mean = posterior_intercept_mean
        self.current_intercept_sd = posterior_intercept_sd


class KnowledgeTrial(StaticTrial):
    time_estimate = 10

    def __init__(self, experiment, node, participant, *args, **kwargs):
        super().__init__(experiment, node, participant, *args, **kwargs)

        self.var.correct = None

    def show_trial(self, experiment, participant):
        question = self.definition["question"]

        page = ModularPage(
            "knowledge_trial",
            Markup(
                f"""
                <p id='question'>{question}</p>
                (Leave empty if you do not know)
                """
            ),
            TextControl(
                block_copy_paste=True,
                bot_response=lambda: oracle.answer(participant.id, self.definition["item_id"])
            ),
            time_estimate=self.time_estimate,
        )

        return page

    def show_feedback(self, experiment, participant):
        return InfoPage(
            Markup("Congratulations, this is correct!")
            if self.var.correct == True
            else Markup("Nice try, but no :(")
        )


class KnowledgeTrialMaker(StaticTrialMaker):
    def __init__(self, *args, setup: str = "adaptive", **kwargs):
        questions = pd.read_csv("static/questions.csv").head(15)

        nodes = [
            StaticNode(
                definition={
                    "item_id": i,
                    "question": question["question"],
                    "answers": question["answers"].split("|")
                }
            )
            for i, question in enumerate(questions.to_dict(orient="records"))
        ]

        super().__init__(*args, **kwargs, nodes=nodes)

        self.setup = setup
        logger.info("Initializing adaptive learner?")
        self.learner = AdaptiveLearner(len(nodes))

    def custom_network_filter(self, candidates, participant):
        if self.setup == "static":
            return candidates

        logger.info("Getting data for participant")
        if participant.id not in self.learner.participants.values():
            logger.info("New participant")
            pid = self.learner.add_participant(participant)
            logger.info(self.learner.participants)
        else:
            pid = self.learner.get_participant(participant)

        candidate_items = []
        for candidate in candidates:
            candidate_items += [node.definition["item_id"] for node in candidate.nodes()]

        # exclude items that are not among the candidate items
        exclude = [item for item in range(self.learner.num_items) if item not in candidate_items]

        # choose best item
        next_node_id = self.learner.get_optimal_test_for_participant(pid, exclude=exclude)

        # recover the retained candidate
        for candidate in candidates:
            if any([node.definition["item_id"] == next_node_id for node in candidate.nodes()]):
                return [candidate]

        return candidates

    def finalize_trial(self, answer, trial, experiment, participant):
        trial.var.correct = trial.answer.lower() in trial.node.definition["answers"]

        logger.info("Finalizing trial, looking for participant")
        if participant.id not in self.learner.participants.values():
            logger.info("New participant")
            pid = self.learner.add_participant(participant)
            logger.info(self.learner.participants)
        else:
            pid = self.learner.get_participant(participant)

        self.learner.administer_response(pid, trial.node.definition["item_id"], trial.var.correct)
        self.learner.update_posterior()

        mu = self.learner.current_theta_means[pid]
        sd = self.learner.current_theta_sds[pid]
        logger.info(f"Posterior participant ability: N({mu:.2f}, {sd:.2f})")

        mu = self.learner.current_difficulty_means[trial.node.definition["item_id"]]
        sd = self.learner.current_difficulty_sds[trial.node.definition["item_id"]]
        logger.info(f"Posterior item difficulty: N({mu:.2f}, {sd:.2f})")
        logger.info(trial.node.definition["question"])
        logger.info(trial.answer)

        super().finalize_trial(answer, trial, experiment, participant)


trial_maker = KnowledgeTrialMaker(
    id_="knowledge",
    trial_class=KnowledgeTrial,
    expected_trials_per_participant=15,
    max_trials_per_participant=15,
    allow_repeated_nodes=False,
    balance_across_nodes=True,
    target_n_participants=1,
    target_trials_per_node=None,
    recruit_mode="n_participants",
    n_repeat_trials=1,
)


class Exp(psynet.experiment.Experiment):
    label = "Adaptive Bayesian testing demo"
    initial_recruitment_size = 1
    test_n_bots = 200

    timeline = Timeline(
        trial_maker,
    )
