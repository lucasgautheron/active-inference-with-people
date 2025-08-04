# pylint: disable=unused-import,abstract-method

import logging

from markupsafe import Markup

import psynet.experiment
from psynet.modular_page import TextControl, ModularPage
from psynet.page import InfoPage, SuccessfulEndPage
from psynet.timeline import Timeline, Response
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

from scipy.stats import entropy

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
        mask = ~np.any(pd.isna(answers), axis=1)
        answers = answers[mask]

        self.answers = [
            {
                "answers": answers[i],
            }
            for i in range(len(answers))
        ]

        self.education = pd.read_csv(
            "static/education.csv"
        )["college"].values[mask]

        logger.info(self.education)

    def answer(self, participant_id: int, item: int):
        return self.answers[participant_id]["answers"][item]

    def college(self, participant_id: int):
        return self.education[participant_id]


oracle = Oracle()


class KnowledgeTrial(StaticTrial):
    time_estimate = 10  # how long it should take to complete each trial, in seconds

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

        self.var.y = None  # correct answer?
        self.var.z = None  # education?

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

        logger.info("Initializing adaptive learner.")

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
            for i, question in enumerate(
                questions.to_dict(orient="records")
            )
        ]

        return nodes

    def thompson_sampling(self, nodes):
        n_samples = 1000

        # Draw Phi
        alpha, beta = 1, 1

        participants = KnowledgeTrial.query.distinct(
            KnowledgeTrial.participant_id
        ).all()

        for participant in participants:
            if participant.var.z == True:
                alpha += 1
            elif participant.var.z == False:
                beta += 1

        Phi = np.random.beta(alpha, beta, n_samples)
        # if DEBUG_MODE:
        #     Phi = np.ones(n_samples) * 0.14

        rewards = dict()
        for node in nodes:
            alpha = np.ones(2)
            beta = np.ones(2)
            for trial in node.viable_trials:
                if trial.var.z is None:
                    continue

                z = 1 if trial.var.z else 0
                if trial.var.y == True:
                    alpha[z] += 1
                elif trial.var.y == False:
                    beta[z] += 1

            phi = np.random.beta(
                alpha[:, np.newaxis],
                beta[:, np.newaxis],
                (2, n_samples),
            )

            rewards[node.id] = entropy(
                [Phi, 1 - Phi], axis=0, base=2
            )

            for z in [False, True]:
                p_z = Phi if z else 1 - Phi
                for y in [False, True]:
                    p_y_given_z = (
                        phi[z * 1] if y else 1 - phi[z * 1]
                    )
                    p_y = (
                        phi[1] * Phi + phi[0] * (1 - Phi)
                        if y
                        else (
                            (1 - phi[1]) * Phi
                            + (1 - phi[0]) * (1 - Phi)
                        )
                    )

                    rewards[node.id] += (
                        p_y_given_z
                        * p_z
                        * np.log2(p_y_given_z * p_z / p_y)
                    )

            assert (rewards[node.id] > -1e-6).all()

        from matplotlib import pyplot as plt
        import seaborn as sns

        if np.random.uniform() > 1:
            cmap = plt.get_cmap("tab10")
            fig, ax = plt.subplots()
            for node in nodes:
                color = cmap(node.definition["item_id"])
                sns.kdeplot(
                    rewards[node.id], ax=ax, color=color
                )
            ax.set_xlim(0, entropy([Phi[0], 1 - Phi[0]]))
            ax.set_ylim(0, 60)
            plt.show()

        best_node = sorted(
            list(rewards.keys()),
            key=lambda node: rewards[node][0],
            reverse=True,
        )[0]

        for node in nodes:
            if node.id == best_node:
                logger.info(node.definition)

        return best_node

    def custom_network_filter(
        self,
        candidates,
        participant,
    ):

        nodes = []
        for candidate in candidates:
            nodes += candidate.nodes()

        next_node_id = self.thompson_sampling(nodes)

        for candidate in candidates:
            if any(
                [
                    node.id == next_node_id
                    for node in candidate.nodes()
                ]
            ):
                return [candidate]

        return candidates

    def finalize_trial(
        self,
        answer,
        trial,
        experiment,
        participant,
    ):
        trial.var.y = (
            trial.answer.lower()
            in trial.node.definition["answers"]
        )

        response = Response.query.filter_by(
            question="formal_education",
            participant_id=participant.id,
        ).one()

        if DEBUG_MODE:
            trial.var.z = bool(
                oracle.college(participant.id)
            )
        else:
            trial.var.z = response.answer in [
                "college",
                "graduate_school",
                "postgraduate_degree_or_higher",
            ]

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
        KnowledgeTrialMaker(),
        SuccessfulEndPage(),
    )
