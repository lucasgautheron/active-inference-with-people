import math
import os

import numpy as np
import pytest

os.environ.setdefault("DALLINGER_NO_EXPERIMENT_PRELOAD", "1")

import experiment as experiment_module
from experiment import (
    AdaptiveTesting,
    AdaptiveTreatment,
    OptimalDesign,
    beta_bernoulli_eig,
)

experiment_module.DEBUG_MODE = False


class StubVar:
    def __init__(self, z=1):
        self.z = z


class StubParticipant:
    def __init__(self, participant_id, z=1):
        self.id = participant_id
        self.var = StubVar(z)


def treatment_data():
    return {
        "participants": {
            1: {"z": 1},
            2: {"z": 0},
        },
        "items": {10: {}, 11: {}, 12: {}},
        "y": {
            100: {
                "value": 1,
                "participant_id": 1,
                "item_id": 10,
            },
        },
    }


def test_beta_bernoulli_eig_uniform_prior():
    eig = float(beta_bernoulli_eig(1.0, 1.0))
    expected = math.log(2.0) - 0.5
    assert eig == pytest.approx(expected, rel=1e-6)


def test_beta_bernoulli_eig_decreases_with_certainty():
    assert beta_bernoulli_eig(1, 1) > beta_bernoulli_eig(
        50, 50
    )


def test_adaptive_treatment_posterior_counts():
    opt = AdaptiveTreatment()
    opt.update_posterior(treatment_data())
    assert opt.alpha[10][1] == 2
    assert opt.beta[10][1] == 1
    assert np.array_equal(opt.alpha[11], np.ones(2))
    assert np.array_equal(opt.beta[11], np.ones(2))


def test_adaptive_treatment_utility_formula():
    opt = AdaptiveTreatment(gamma=0.1)
    opt.alpha = {
        10: np.array([1.0, 3.0]),
        11: np.array([3.0, 1.0]),
    }
    opt.beta = {
        10: np.array([1.0, 1.0]),
        11: np.array([1.0, 1.0]),
    }
    participant = StubParticipant(1, z=1)
    utility = opt.expected_utility(
        [10, 11], participant, data={}
    )
    assert utility[10] == pytest.approx(
        0.1 * (3 / 4 - 1 / 2)
    )
    assert utility[11] == pytest.approx(
        0.1 * (1 / 2 - 3 / 4)
    )


def test_adaptive_treatment_gamma_zero_selects_max_eig():
    data = {
        "participants": {1: {"z": 1}, 2: {"z": 0}},
        "items": {10: {}, 11: {}},
        "y": {
            1: {
                "value": 1,
                "participant_id": 1,
                "item_id": 11,
            },
            2: {
                "value": 1,
                "participant_id": 1,
                "item_id": 11,
            },
            3: {
                "value": 0,
                "participant_id": 1,
                "item_id": 11,
            },
            4: {
                "value": 0,
                "participant_id": 1,
                "item_id": 11,
            },
        },
    }
    opt = AdaptiveTreatment(gamma=0.0)
    node, p = opt.get_optimal_node(
        [10, 11], StubParticipant(1, z=1), data
    )
    assert node == 10
    assert set(p) == {0, 1}
    assert pytest.approx(p[0] + p[1]) == 1.0


def test_adaptive_treatment_large_gamma_selects_max_utility():
    data = {
        "participants": {1: {"z": 1}, 2: {"z": 0}},
        "items": {10: {}, 11: {}},
        "y": {
            1: {
                "value": 1,
                "participant_id": 1,
                "item_id": 11,
            },
            2: {
                "value": 1,
                "participant_id": 1,
                "item_id": 11,
            },
            3: {
                "value": 0,
                "participant_id": 2,
                "item_id": 11,
            },
            4: {
                "value": 0,
                "participant_id": 2,
                "item_id": 11,
            },
        },
    }
    opt = AdaptiveTreatment(gamma=10.0)
    node, _ = opt.get_optimal_node(
        [10, 11], StubParticipant(1, z=1), data
    )
    assert node == 11


def test_adaptive_testing_posterior_is_finite():
    data = {
        "participants": {1: {"z": None}, 2: {"z": None}},
        "items": {10: {}, 11: {}},
        "y": {
            1: {
                "value": 1.0,
                "participant_id": 1,
                "item_id": 10,
            },
            2: {
                "value": 0.0,
                "participant_id": 2,
                "item_id": 11,
            },
        },
    }
    opt = AdaptiveTesting(num_steps=20, svi_lr=0.05)
    opt.update_posterior(data)
    assert torch_all_finite(opt.theta_means)
    assert torch_all_finite(opt.theta_sds)
    assert torch_all_finite(opt.difficulty_means)
    assert torch_all_finite(opt.difficulty_sds)
    assert torch_all_finite(opt.intercept_mean)
    assert torch_all_finite(opt.intercept_sd)
    assert (opt.theta_sds > 0).all()
    assert (opt.difficulty_sds > 0).all()


def test_adaptive_testing_get_optimal_node(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    data = {
        "participants": {1: {"z": None}},
        "items": {10: {}, 11: {}},
        "y": {
            1: {
                "value": 1.0,
                "participant_id": 1,
                "item_id": 10,
            },
        },
    }
    opt = AdaptiveTesting(
        num_steps=8,
        num_samples=8,
        final_num_samples=32,
        epsilon=0.0,
        svi_lr=0.05,
    )
    node, p = opt.get_optimal_node(
        [10, 11], StubParticipant(1), data
    )
    assert node in (10, 11)
    assert set(p) == {0, 1}
    assert 0.0 <= p[1] <= 1.0


def test_adaptive_testing_prior_eig_exceeds_epsilon():
    data = {
        "participants": {1: {"z": None}},
        "items": {10: {}, 11: {}},
        "y": {},
    }
    opt = AdaptiveTesting(
        num_steps=5,
        num_samples=16,
        final_num_samples=64,
        epsilon=0.04,
    )
    opt.update_posterior(data)
    eig = opt.expected_information_gain(
        [10, 11], StubParticipant(1), data
    )
    assert max(eig.values()) > 0.04
    node, p = opt.get_optimal_node(
        [10, 11], StubParticipant(1), data
    )
    assert node in (10, 11)
    assert set(p) == {0, 1}
    opt = AdaptiveTesting(epsilon=0.04)
    assert opt.should_stop({10: 0.01, 11: 0.02})
    assert not opt.should_stop({10: 0.05, 11: 0.02})


def test_plug_and_play_loop():
    class Dummy(OptimalDesign):
        def expected_information_gain(
            self, candidates, participant, data
        ):
            return {d: float(d) for d in candidates}

    node, p = Dummy().get_optimal_node(
        [1, 3, 2], StubParticipant(1), data={}
    )
    assert node == 3
    assert p == {0: 0.5, 1: 0.5}


def torch_all_finite(value):
    import torch

    return bool(torch.isfinite(torch.as_tensor(value)).all())
