# pylint: disable=unused-import,abstract-method

import logging
import random
import time
import json

from markupsafe import Markup

import psynet.experiment
from psynet.modular_page import TextControl, ModularPage
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Timeline, CodeBlock, ModuleState
from psynet.participant import Participant
from psynet.trial.main import Trial
from psynet.trial.static import (
    StaticNode,
    StaticTrial,
    StaticTrialMaker,
)
from psynet.consent import MainConsent
from psynet.demography.general import (
    Age,
    FormalEducation,
)
from psynet.utils import log_time_taken

from dallinger import db

import torch
import pyro
import pyro.distributions as dist
import numpy as np
from pyro.contrib.oed.eig import marginal_eig
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_mean
from pyro.optim import Adam
from scipy.special import digamma
from scipy.stats import norm

import pandas as pd

DEBUG_PLOTS = False
SETUP = "adaptive"
RECRUITER = "hotair"
DURATION_ESTIMATE = 60 + 30 * 20  # in seconds

assert SETUP in ["adaptive", "oracle"]
assert RECRUITER in ["hotair", "prolific", "cap-recruiter"]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Oracle:
    """
    Oracle for simulating the experiment
    using real human data
    """

    def __init__(self, domains):
        self.domains = domains
        self.answers = None
        self.education = None

    def _load(self):
        if self.answers is not None:
            return

        answers = pd.read_csv(
            "output/KnowledgeTrial_oracle_treatment.csv"
        )
        answers["domain"] = (answers["node_id"] - 1) // 15
        answers = answers[answers["domain"].isin(self.domains)]

        logger.info(answers["answer"])

        participants = pd.read_csv(
            "output/Participant_oracle_treatment.csv"
        )
        participants = participants[
            participants["progress"] == 1
        ]
        participants.reset_index(inplace=True)
        participants["new_participant_id"] = (
            participants.index.values + 1
        )

        logger.info(
            participants[
                participants["new_participant_id"] == 103
            ]
        )

        answers = answers.merge(
            participants[["id", "new_participant_id"]],
            how="inner",
            left_on="participant_id",
            right_on="id",
        )

        logger.info(
            answers[answers["new_participant_id"] == 103]
        )

        answers["answer"].fillna("", inplace=True)
        self.answers = {
            (
                answer["new_participant_id"],
                answer["question"],
            ): answer["answer"]
            for answer in answers.to_dict(orient="records")
        }
        self.education = {
            participant["new_participant_id"]: participant[
                "z"
            ]
            for participant in participants.to_dict(
                orient="records"
            )
        }

        logger.info("Oracle data:")
        logger.info(answers.shape)

    def answer(self, participant_id: int, item_id: int):
        self._load()
        return self.answers[(participant_id, item_id)]

    def college(self, participant_id: int):
        self._load()
        return self.education[participant_id]


oracle = Oracle(domains=[0, 1])


def beta_bernoulli_eig(alpha, beta):
    """Exact EIG for a Bernoulli likelihood with a Beta posterior."""
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    n = alpha + beta
    mu = alpha / n
    predictive_entropy = -mu * np.log(mu) - (
        1 - mu
    ) * np.log1p(-mu)
    expected_conditional_entropy = (
        digamma(n + 1)
        - mu * digamma(alpha + 1)
        - (1 - mu) * digamma(beta + 1)
    )
    return predictive_entropy - expected_conditional_entropy


class OptimalDesign:
    """Active inference loop shared by every optimizer."""

    def update_posterior(self, data):
        return None

    def expected_information_gain(
        self, candidates, participant, data
    ):
        raise NotImplementedError()

    def expected_utility(
        self, candidates, participant, data
    ):
        return {d: 0.0 for d in candidates}

    def predictive_outcome(
        self, candidates, participant, data
    ):
        return {d: 0.5 for d in candidates}

    def should_stop(self, eig):
        return False

    def get_optimal_node(
        self, candidates, participant, data
    ):
        self.update_posterior(data)
        eig = self.expected_information_gain(
            candidates, participant, data
        )
        utility = self.expected_utility(
            candidates, participant, data
        )
        if self.should_stop(eig):
            logger.info("Early stopping")
            return None, None

        scores = {
            d: eig[d] + utility[d] for d in candidates
        }
        maximum = max(scores.values())
        ties = [
            d
            for d in candidates
            if abs(scores[d] - maximum) < 1e-5
        ]
        d_hat = random.choice(ties)
        p_y = self.predictive_outcome(
            candidates, participant, data
        )[d_hat]
        return d_hat, {0: 1.0 - p_y, 1: float(p_y)}


class AdaptiveTesting(OptimalDesign):
    def __init__(
        self,
        num_steps=400,
        num_samples=400,
        final_num_samples=10000,
        epsilon=0.04,
        svi_lr=0.02,
        start_lr=0.1,
        end_lr=0.001,
    ):
        logger.debug("Initializing adaptive learner.")

        # Independent sites so AutoNormal matches the prior family.
        # logit = θ − δ + b implements the paper 1PL δ ~ N(b, 1).
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
        self.intercept_sd = torch.tensor(1.0)

        self.num_steps = num_steps
        self.num_samples = num_samples
        self.final_num_samples = final_num_samples
        self.epsilon = epsilon
        self.svi_lr = svi_lr
        self.start_lr = start_lr
        self.end_lr = end_lr
        self._p_y = {}

    def _model(self, participants, items):
        """(1) Observation model: 1PL item-response model."""
        thetas = pyro.sample(
            "thetas",
            dist.Normal(
                self.prior_mean_theta,
                self.prior_sd_theta,
            )
            .expand([self.num_participants])
            .to_event(1),
        )
        difficulties = pyro.sample(
            "difficulties",
            dist.Normal(
                self.prior_mean_difficulty,
                self.prior_sd_difficulty,
            )
            .expand([self.num_items])
            .to_event(1),
        )
        intercept = pyro.sample(
            "intercept",
            dist.Normal(
                self.prior_mean_intercept,
                self.prior_sd_intercept,
            ),
        )
        logit_p = (
            thetas[participants.long()]
            - difficulties[items.long()]
            + intercept
        )
        pyro.sample(
            "y",
            dist.Bernoulli(logits=logit_p).to_event(1),
        )

    def _make_design_model(self, target_participant):
        """(2) Simulation model for a candidate item."""

        def model(design):
            with pyro.plate_stack(
                "plate", design.shape[:-1]
            ):
                theta = pyro.sample(
                    "theta",
                    dist.Normal(
                        self.theta_means[
                            target_participant
                        ],
                        self.theta_sds[target_participant],
                    ),
                ).unsqueeze(-1)
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
                pyro.sample(
                    "y",
                    dist.Bernoulli(
                        logits=(theta - difficulties)
                        + intercept
                    ).to_event(1),
                )

        return model

    def _marginal_guide(
        self,
        design,
        observation_labels,
        target_labels,
    ):
        q_logit = pyro.param(
            "q_logit",
            torch.zeros(design.shape[-2:]),
        )
        pyro.sample(
            "y",
            dist.Bernoulli(logits=q_logit).to_event(1),
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
        self.intercept_mean = self.prior_mean_intercept.clone()
        self.intercept_sd = self.prior_sd_intercept.clone()

    def _loc_scale(self, name):
        loc, scale = self.guide._get_loc_and_scale(name)
        return loc.detach().clone(), scale.detach().clone()

    def update_posterior(self, data):
        """(3) Mean-field Gaussian posterior via AutoNormal."""
        self.participant_index = {
            pid: i
            for i, pid in enumerate(data["participants"])
        }
        self.item_index = {
            iid: i for i, iid in enumerate(data["items"])
        }
        self.init_parameters(
            len(self.participant_index),
            len(self.item_index),
        )

        participant_idx = torch.tensor(
            [
                self.participant_index[obs["participant_id"]]
                for obs in data["y"].values()
            ],
            dtype=torch.long,
        )
        item_idx = torch.tensor(
            [
                self.item_index[obs["item_id"]]
                for obs in data["y"].values()
            ],
            dtype=torch.long,
        )
        responses = torch.tensor(
            [float(obs["value"]) for obs in data["y"].values()],
            dtype=torch.float,
        )

        pyro.clear_param_store()
        conditioned_model = pyro.condition(
            self._model, {"y": responses}
        )
        self.guide = AutoNormal(
            conditioned_model,
            init_loc_fn=init_to_mean,
            init_scale=1.0,
        )
        svi = SVI(
            conditioned_model,
            self.guide,
            Adam({"lr": self.svi_lr}),
            loss=Trace_ELBO(),
        )
        for i in range(self.num_steps):
            elbo = svi.step(participant_idx, item_idx)
            if i % 100 == 0:
                logger.info(
                    f"  Iteration {i}, ELBO: {elbo:.3f}"
                )

        self.theta_means, self.theta_sds = self._loc_scale(
            "thetas"
        )
        (
            self.difficulty_means,
            self.difficulty_sds,
        ) = self._loc_scale("difficulties")
        self.intercept_mean, self.intercept_sd = (
            self._loc_scale("intercept")
        )
        logger.debug("Posterior update completed")

    def expected_information_gain(
        self, candidates, participant, data
    ):
        pyro.clear_param_store()
        design_model = self._make_design_model(
            self.participant_index[participant.id]
        )
        candidate_designs = torch.tensor(
            [self.item_index[item] for item in candidates],
            dtype=torch.float,
        ).unsqueeze(-1)

        optimizer = pyro.optim.ExponentialLR(
            {
                "optimizer": torch.optim.Adam,
                "optim_args": {"lr": self.start_lr},
                "gamma": (self.end_lr / self.start_lr)
                ** (1 / max(self.num_steps, 1)),
            }
        )
        eig = marginal_eig(
            design_model,
            candidate_designs,
            "y",
            ["theta", "difficulties", "intercept"],
            num_samples=self.num_samples,
            num_steps=self.num_steps,
            guide=self._marginal_guide,
            optim=optimizer,
            final_num_samples=self.final_num_samples,
        )
        p_y = (
            torch.special.expit(pyro.param("q_logit"))
            .detach()
            .reshape(-1)
            .numpy()
        )
        eig_np = eig.detach().reshape(-1).numpy()
        self._p_y = {
            candidate: float(p_y[idx])
            for idx, candidate in enumerate(candidates)
        }
        eig_dict = {
            candidate: float(eig_np[idx])
            for idx, candidate in enumerate(candidates)
        }

        logger.info(
            "EIG max=%.4f epsilon=%.4f"
            % (max(eig_dict.values()), self.epsilon)
        )

        if DEBUG_PLOTS:
            self._debug_plot(
                candidates, participant, eig_np, p_y
            )

        return eig_dict

    def predictive_outcome(
        self, candidates, participant, data
    ):
        return self._p_y

    def should_stop(self, eig):
        """(4) Stop when max EIG falls below epsilon."""
        return max(eig.values()) < self.epsilon

    def _debug_plot(
        self, candidates, participant, eig, p_y
    ):
        from matplotlib import pyplot as plt

        entropy = -p_y * np.log2(p_y) - (1 - p_y) * np.log2(
            1 - p_y
        )
        x = np.linspace(-3, +3, 200)
        idx = self.participant_index[participant.id]
        plt.plot(
            x,
            norm.pdf(
                x,
                loc=self.theta_means[idx],
                scale=self.theta_sds[idx],
            ),
            color="black",
            label=r"$\theta$(participant)",
        )
        cmap = plt.get_cmap("tab10")
        for i, item in enumerate(candidates):
            color = cmap(i % 10)
            item_i = self.item_index[item]
            plt.plot(
                x,
                norm.pdf(
                    x,
                    loc=self.difficulty_means[item_i]
                    - self.intercept_mean,
                    scale=np.sqrt(
                        self.difficulty_sds[item_i] ** 2
                        + self.intercept_sd**2
                    ),
                ),
                alpha=0.2,
                color=color,
            )
            plt.scatter(
                [
                    self.difficulty_means[item_i]
                    - self.intercept_mean
                ],
                [eig[i]],
                facecolors="none",
                edgecolors=color,
                marker="s",
                label="EIG" if i == 0 else None,
            )
            plt.scatter(
                [
                    self.difficulty_means[item_i]
                    - self.intercept_mean
                ],
                [entropy[i]],
                color=color,
                label="$H(y)$" if i == 0 else None,
            )
        plt.axhline(self.epsilon, label=r"$\varepsilon$")
        plt.xlim(-3, 3)
        plt.ylim(0, 1)
        plt.legend()
        plt.savefig(f"output/test_{participant.id}.png")
        plt.clf()


class AdaptiveTreatment(OptimalDesign):
    def __init__(self, gamma=0.1):
        self.gamma = gamma
        self.alpha = {}
        self.beta = {}

    def update_posterior(self, data):
        """Exact Beta posterior by conjugacy."""
        self.alpha = {
            item: np.ones(2) for item in data["items"]
        }
        self.beta = {
            item: np.ones(2) for item in data["items"]
        }
        for obs in data["y"].values():
            z = data["participants"][obs["participant_id"]][
                "z"
            ]
            if z is None:
                continue
            z = int(z)
            item = obs["item_id"]
            if obs["value"]:
                self.alpha[item][z] += 1
            else:
                self.beta[item][z] += 1

    def expected_information_gain(
        self, candidates, participant, data
    ):
        z_i = int(participant.var.z)
        return {
            d: float(
                beta_bernoulli_eig(
                    self.alpha[d][z_i], self.beta[d][z_i]
                )
            )
            for d in candidates
        }

    def expected_utility(
        self, candidates, participant, data
    ):
        """(5) U(d) = γ (μ_{d,1} − μ_{d,0}), p(z)=1/2."""
        return {
            d: float(
                self.gamma
                * (
                    self.alpha[d][1]
                    / (self.alpha[d][1] + self.beta[d][1])
                    - self.alpha[d][0]
                    / (self.alpha[d][0] + self.beta[d][0])
                )
            )
            for d in candidates
        }

    def predictive_outcome(
        self, candidates, participant, data
    ):
        z_i = int(participant.var.z)
        return {
            d: float(
                self.alpha[d][z_i]
                / (self.alpha[d][z_i] + self.beta[d][z_i])
            )
            for d in candidates
        }


class KnowledgeTrial(StaticTrial):
    time_estimate = (
        25  # how long it should take to complete each trial, in seconds
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

        # Relevant participant metadata
        self.var.z = None

        # Posterior predictive probability of given answer
        self.var.p = None

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
                    question,
                ),
            ),
            time_estimate=self.time_estimate,
        )

        return page

    def score_answer(self, answer, definition):
        return 1 if answer.lower().strip() in definition["answers"] else 0

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
        domains,
        use_participant_data,
        *args,
        **kwargs,
    ):
        """
        Initialize the trial maker
        with the list of all possibles challenges
        """
        nodes = self.load_nodes(domains)

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
        self.optimizer = (
            optimizer_class() if optimizer_class is not None else None
        )
        self.use_participant_data = use_participant_data

    def load_nodes(self, domains: list):
        questions = pd.read_csv("static/questions.csv")
        questions["domain"] = questions["id"] // 15
        questions = questions[questions["domain"].isin(domains)]
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

    @log_time_taken
    def prior_data(self, experiment):
        data = {
            "participants": dict(),
            "items": dict(),
            "y": dict(),
        }

        start = time.time()
        participants = (
            db.session.query(Participant)
            .join(Participant._module_states)
            .filter(
                ModuleState.module_id == self.id,
                ModuleState.started == True,
            )
            .distinct()
            .all()
        )

        data["participants"] = {
            participant.id: {
                "z": (
                    participant.var.get("z", None)
                    if self.use_participant_data
                    else None
                ),
            }
            for participant in participants
        }
        logger.info(
            f"Processing participants: {time.time() - start:.3f}s"
        )

        start = time.time()
        networks = self.network_class.query.filter_by(
            trial_maker_id=self.id,
        ).all()
        nodes = [network.head for network in networks]
        data["items"] = {node.id: {} for node in nodes}
        logger.info(f"Nodes query: {time.time() - start:.3f}s")

        start = time.time()
        trials = Trial.query.filter(
            Trial.failed == False,
            Trial.finalized == True,
            Trial.is_repeat_trial == False,
            Trial.trial_maker_id == self.id,
            Trial.score != None,
        ).all()
        logger.info(f"Trials query: {time.time() - start:.3f}s")

        start = time.time()
        data["y"] = {
            trial.id: {
                "value": trial.score,
                "participant_id": trial.participant_id,
                "item_id": trial.node_id,
            }
            for trial in trials
        }
        logger.info(
            f"Processing observations: {time.time() - start:.3f}s"
        )

        return data

    @log_time_taken
    def find_nodes(self, participant, experiment):
        nodes = super().find_nodes(participant, experiment)
        if not isinstance(nodes, list) or self.optimizer is None:
            return nodes

        candidates = {node.id: node for node in nodes}

        data = self.prior_data(experiment)
        next_node, p = self.optimizer.get_optimal_node(
            list(candidates.keys()), participant, data
        )

        participant.var.set("p_y", p)

        if next_node is None:
            return "exit"

        return [candidates[next_node]]

    def finalize_trial(
        self,
        answer,
        trial,
        experiment,
        participant,
    ):
        trial.var.y = (
            trial.answer.lower().strip() in trial.node.definition["answers"]
        )

        trial.var.z = (
            int(trial.participant.var.z) if self.use_participant_data else None
        )
        trial.var.p = participant.var.get("p_y", None)

        logger.info(trial.var)

        super().finalize_trial(
            answer,
            trial,
            experiment,
            participant,
        )


def get_prolific_settings(experiment_duration):
    with open("qualification_prolific_en.json", "r") as f:
        qualification = json.dumps(json.load(f))

    return {
        "recruiter": "prolific",
        "base_payment": 9 * DURATION_ESTIMATE / 60 / 60,
        "prolific_estimated_completion_minutes": DURATION_ESTIMATE / 60,
        "prolific_recruitment_config": qualification,
        "auto_recruit": False,
        "wage_per_hour": 0,
        "currency": "$",
        "show_reward": False,
    }


def get_cap_settings(experiment_duration):
    raise {"wage_per_hour": 12}


recruiter_settings = None
if RECRUITER == "prolific":
    recruiter_settings = get_prolific_settings(DURATION_ESTIMATE)
elif RECRUITER == "cap-recruiter":
    recruiter_settings = get_cap_settings(DURATION_ESTIMATE)


class Exp(psynet.experiment.Experiment):
    label = "Active inference for adaptive experiments"
    test_n_bots = 200
    test_mode = "serial"

    config = {
        "recruiter": RECRUITER,
        "wage_per_hour": 0,
        "auto_recruit": False,
        "show_reward": False,
        "initial_recruitment_size": 3,
    }

    if RECRUITER != "hotair":
        config.update(**recruiter_settings)

    timeline = Timeline(
        MainConsent(),
        Age(),
        FormalEducation(),
        InfoPage(
            Markup(
                f"<h3>Before we begin...</h3>"
                f"<div style='margin: 10px;'>You will be presented with a series of trivia questions, such as \"Who was the first man to step on the moon?\".</div>"
                f"<div style='margin: 10px;'>If you do not know the answer to a question, just skip to the next question.</div>"
                f"<div style='margin: 10px;'>Please do <i>not</i> write your answer as sentences. For instance, if the question is: what is the current year? Please just answer '2025'. Do <i>NOT</i> answer, say, 'The current year is 2025'. </div>"
            ),
            time_estimate=15,
        ),
        CodeBlock(
            lambda participant: participant.var.set(
                "z",
                (
                    participant.answer
                    in [
                        "college",
                        "graduate_school",
                        "postgraduate_degree_or_higher",
                    ]
                )
                * 1,
            )
        ),
        KnowledgeTrialMaker(
            id_="optimal_treatment",
            optimizer_class=(
                AdaptiveTreatment if SETUP == "adaptive" else None
            ),
            domains=[0, 1],
            use_participant_data=True,
            expected_trials_per_participant=(
                5 if SETUP == "adaptive" else 30
            ),
            max_trials_per_participant=(
                5 if SETUP == "adaptive" else 30
            ),
        ),
        KnowledgeTrialMaker(
            id_="optimal_test",
            optimizer_class=(
                AdaptiveTesting if SETUP == "adaptive" else None
            ),
            domains=[0],
            use_participant_data=False,
            expected_trials_per_participant=15,
            max_trials_per_participant=15,
        ),
        SuccessfulEndPage(),
    )
