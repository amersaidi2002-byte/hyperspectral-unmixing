import numpy as np
import pytest
from scipy.optimize import minimize

from spectral_unmixing.algorithms import (
    fcls,
    grid_laplacian,
    project_simplex,
    spa,
    spatial_energy,
    unmix,
)
from spectral_unmixing.data import synthetic_scene


def test_euclidean_simplex_projection():
    v = np.array([[0.2, 0.3, 0.7], [2, -0.5, 0.3], [-1, -1, -1]])
    expected = [[2 / 15, 7 / 30, 19 / 30], [1, 0, 0], [1 / 3, 1 / 3, 1 / 3]]
    np.testing.assert_allclose(project_simplex(v), expected, atol=1e-12)
    np.testing.assert_allclose(project_simplex(v + 20), expected, atol=1e-12)


@pytest.mark.parametrize("v", [np.array([[np.nan, 0]]), np.empty((2, 0)), np.ones(2)])
def test_simplex_rejects_invalid_values(v):
    with pytest.raises(ValueError):
        project_simplex(v)


def test_fcls_against_independent_optimizer():
    rng = np.random.default_rng(24)
    e, x = rng.uniform(0.1, 1, (3, 8)), rng.uniform(0.05, 1.2, (8, 8))
    estimated, diagnostics = fcls(x, e)
    assert diagnostics["converged"]
    for spectrum, a in zip(x, estimated, strict=True):
        reference = minimize(
            lambda z, s=spectrum: 0.5 * np.sum((z @ e - s) ** 2),
            np.ones(3) / 3,
            jac=lambda z, s=spectrum: (z @ e - s) @ e.T,
            bounds=[(0, 1)] * 3,
            constraints={
                "type": "eq",
                "fun": lambda z: z.sum() - 1,
                "jac": lambda z: np.ones(3),
            },
            method="SLSQP",
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        assert reference.success
        np.testing.assert_allclose(a, reference.x, atol=2e-6)


def test_fcls_recovers_exact_fractions():
    scene = synthetic_scene(size=12, bands=24, snr_db=None)
    a, info = fcls(scene.spectra, scene.endmembers)
    assert info["converged"]
    np.testing.assert_allclose(a, scene.abundances.reshape(-1, 3), atol=2e-6)


def test_fcls_is_not_normalized_nnls():
    a, _ = fcls(np.array([[2.0, 0.2]]), np.eye(2))
    np.testing.assert_allclose(a, [[1, 0]], atol=1e-10)


def test_spa_recovers_pure_pixels():
    e = np.array([[1.0, 0.1, 0.2], [0.2, 1.0, 0.1], [0.1, 0.2, 1.0]])
    a = np.vstack([np.eye(3), [0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])
    estimated, selected = spa(a @ e, 3)
    assert set(selected) == {0, 1, 2}
    assert {tuple(v) for v in estimated} == {tuple(v) for v in e}


def test_spa_rejects_rank_deficiency():
    with pytest.raises(ValueError, match="independent"):
        spa(np.ones((6, 5)), 3)


def test_laplacian_gradient_and_boundary():
    rng = np.random.default_rng(7)
    a, direction = rng.normal(size=(6, 2)), rng.normal(size=(6, 2))
    eps = 1e-6
    difference = (
        spatial_energy(a + eps * direction, (2, 3))
        - spatial_energy(a - eps * direction, (2, 3))
    ) / (2 * eps)
    np.testing.assert_allclose(
        difference, 2 * np.sum(grid_laplacian(a, (2, 3)) * direction), rtol=1e-8
    )
    impulse = np.zeros((6, 1))
    impulse[2] = 1
    assert grid_laplacian(impulse, (2, 3))[3, 0] == 0


@pytest.mark.parametrize("method", ["spa-fcls", "simplex-nmf", "spatial-nmf"])
def test_constraints_and_descent(method):
    result = unmix(synthetic_scene(size=12, bands=24).cube, 3, method, max_iter=40)
    assert result.abundances.min() >= 0 and result.endmembers.min() >= 0
    np.testing.assert_allclose(result.abundances.sum(axis=1), 1, atol=1e-12)
    assert np.all(
        np.diff([v["objective"] for v in result.diagnostics.get("history", [])]) <= 1e-12
    )


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        unmix(synthetic_scene(size=12, bands=16).cube, 3, "magic")
