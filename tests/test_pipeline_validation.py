import pytest
from eval.pipeline.validation import _cohens_kappa


def test_cohens_kappa_perfect_agreement():
    # observed = 1.0, expected = 0.5 -> (1 - 0.5) / (1 - 0.5) = 1.0
    kappa = _cohens_kappa(1.0, 0.5)
    assert kappa == 1.0


def test_cohens_kappa_no_agreement():
    # observed = 0.0, expected = 0.5 -> (0 - 0.5) / (1 - 0.5) = -1.0
    kappa = _cohens_kappa(0.0, 0.5)
    assert kappa == -1.0


def test_cohens_kappa_expected_one():
    # Should safely return 1.0 to avoid divide by zero
    kappa = _cohens_kappa(1.0, 1.0)
    assert kappa == 1.0


def test_cohens_kappa_chance_agreement():
    # observed = expected -> 0.0
    kappa = _cohens_kappa(0.5, 0.5)
    assert kappa == 0.0
