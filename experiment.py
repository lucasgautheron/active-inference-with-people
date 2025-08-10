# pylint: disable=unused-import,abstract-method

import logging

from markupsafe import Markup

import psynet.experiment
from psynet.modular_page import TextControl, ModularPage
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Timeline, Response, CodeBlock
from psynet.trial.static import (
    StaticNode,
    StaticTrial,
    StaticTrialMaker,
)
from psynet.consent import MainConsent
from psynet.demography.general import (
    FormalEducation,
)

import numpy as np
from pyro.optim import Adam

from scipy.stats import beta as beta_dist
from scipy.special import softmax

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

    def __init__(self):
        answers = pd.read_csv("static/answers.csv")
        answers = np.stack(answers.values)[:, :15]
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


oracle = Oracle()


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

        self.var.y = None  # correct answer
        self.var.z = None  # participant education-level

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
            expected_trials_per_participant=5,
            # number of questions administered
            # to each participant (maximum)
            max_trials_per_participant=5,
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

    def get_optimal_treatment(self, networks_ids, participant, data):
        z_i = participant.var.z

        n_samples = 1000

        rewards = dict()
        eig = dict()
        utility = dict()

        alphas = dict()
        betas = dict()

        z_participants = np.array(
            [participant["z"] for participant in data["participants"].values()]
        )

        alpha_z = 1 + np.sum(z_participants == 1)
        beta_z = 1 + np.sum(z_participants == 0)
        p_z = alpha_z / (alpha_z + beta_z)

        for network_id in networks_ids:
            alpha = np.ones(2)
            beta = np.ones(2)

            for trial_id, trial in data["networks"][network_id].items():
                if trial["y"] == True:
                    alpha[trial["z"]] += 1
                elif trial["y"] == False:
                    beta[trial["z"]] += 1

            alphas[network_id] = alpha
            betas[network_id] = beta

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

            gamma = 0.1
            U = np.mean(
                np.log(
                    p_z * np.exp(gamma * (y[1] - (1 - y[1])))
                    + (1 - p_z) * np.exp(gamma * ((1 - y[0]) - y[0])),
                )
            )

            rewards[network_id] = EIG + U
            eig[network_id] = EIG
            utility[network_id] = U

        from matplotlib import pyplot as plt

        if np.random.uniform() > 1 and len(networks_ids) == 15:
            cmap = plt.get_cmap("tab10")
            fig, ax = plt.subplots()
            for network_id in networks_ids:
                color = cmap(network_id % 10)
                x = np.linspace(0, 1, 100)
                ax.plot(
                    x,
                    beta_dist.pdf(
                        x,
                        a=alphas[network_id][0],
                        b=betas[network_id][0],
                    ),
                    color=color,
                    label=rewards[network_id],
                )
                ax.plot(
                    x,
                    beta_dist.pdf(
                        x,
                        a=alphas[network_id][1],
                        b=betas[network_id][1],
                    ),
                    color=color,
                    ls="dashed",
                )
            plt.legend()
            plt.show()

        best_network = sorted(
            list(rewards.keys()),
            key=lambda network_id: rewards[network_id],
            reverse=True,
        )[0]

        if len(networks_ids) == 15:
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

    def prior_data(self, experiment):
        data = {"networks": dict(), "participants": dict()}

        networks = self.network_class.query.filter_by(
            trial_maker_id=self.id, full=False, failed=False
        )

        for network in networks:
            data["networks"][network.id] = dict()

            for trial in network.head.viable_trials:
                y = trial.get_y()

                data["networks"][network.id][trial.id] = {
                    "y": y,
                    "z": trial.participant.var.z,
                    "participant_id": trial.participant.id,
                }

                if trial.participant.id not in data["participants"]:
                    data["participants"][trial.participant.id] = {
                        "z": trial.participant.var.z,
                    }

        return data

    def prioritize_networks(self, networks, participant, experiment):
        return networks

        candidates = {network.id: network for network in networks}

        data = self.prior_data(experiment)

        next_network = self.get_optimal_treatment(
            list(candidates.keys()), participant, data
        )

        return [candidates[next_network]]

    def finalize_trial(
        self,
        answer,
        trial,
        experiment,
        participant,
    ):
        trial.var.y = trial.answer.lower() in trial.node.definition["answers"]

        trial.var.z = bool(trial.participant.var.z)

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
        KnowledgeTrialMaker(),
        SuccessfulEndPage(),
    )
