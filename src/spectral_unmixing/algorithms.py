"""Pure-pixel selection, fully constrained least squares and constrained NMF.

Matrices use pixels x bands, fractions use pixels x components, and component
spectra use components x bands. Fitting functions never accept reference labels.
"""

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np


@dataclass
class UnmixingResult:
    endmembers: np.ndarray
    abundances: np.ndarray
    diagnostics: dict = field(default_factory=dict)


def project_simplex(values):
    """Euclidean row projection onto {a >= 0, sum(a) = 1}, by sorted thresholds."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] == 0 or not np.isfinite(v).all():
        raise ValueError("Expected a finite matrix with at least one component.")
    ordered = np.sort(v, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1
    ranks = np.arange(1, v.shape[1] + 1)
    rho = np.sum(ordered - cumulative / ranks > 0, axis=1) - 1
    theta = cumulative[np.arange(len(v)), rho] / (rho + 1)
    return np.maximum(v - theta[:, None], 0)


def _validate_spectra(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all() or x.min() < 0:
        raise ValueError("Expected finite nonnegative pixels x bands spectra.")
    return x


def spa(x, n_components):
    """SPA on L1-normalized spectra; return selected pixels in original units."""
    x = _validate_spectra(x)
    if not 1 <= n_components <= min(x.shape):
        raise ValueError("Invalid component count.")
    residual = x / np.maximum(x.sum(axis=1, keepdims=True), 1e-15)
    original_norm = float(np.linalg.norm(residual, axis=1).max())
    selected = []
    for _ in range(n_components):
        norms = np.einsum("ij,ij->i", residual, residual)
        norms[selected] = -1
        index = int(np.argmax(norms))
        norm = np.sqrt(max(norms[index], 0))
        if norm <= 1e-10 * max(original_norm, 1):
            raise ValueError("Spectra contain too few independent components for SPA.")
        direction = residual[index] / norm
        selected.append(index)
        residual -= (residual @ direction)[:, None] * direction
    return x[selected].copy(), selected


def fcls(x, endmembers, max_iter=3000, tolerance=1e-8):
    """Solve .5||AE-X||² on simplex rows using objective-restarted FISTA."""
    x, e = _validate_spectra(x), _validate_spectra(endmembers)
    if x.shape[1] != e.shape[1] or max_iter < 1 or tolerance <= 0:
        raise ValueError("Invalid FCLS dimensions or settings.")
    gram, cross = e @ e.T, x @ e.T
    lipschitz = float(np.linalg.eigvalsh(gram)[-1])
    if lipschitz <= 1e-15:
        raise ValueError("Endmembers must contain a nonzero spectrum.")
    a = np.full((len(x), len(e)), 1 / len(e))
    z, momentum = a.copy(), 1.0
    converged, residual = False, float("inf")

    def objective(q):
        return 0.5 * np.sum((q @ gram) * q) - np.sum(q * cross)

    previous = float(objective(a))
    for iteration in range(1, max_iter + 1):
        candidate = project_simplex(z - (z @ gram - cross) / lipschitz)
        current = float(objective(candidate))
        if current > previous + 1e-10:
            z, momentum = a, 1.0
            candidate = project_simplex(z - (z @ gram - cross) / lipschitz)
            current = float(objective(candidate))
        next_momentum = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        z = candidate + ((momentum - 1) / next_momentum) * (candidate - a)
        a, momentum, previous = candidate, next_momentum, current
        if iteration % 20 == 0 or iteration == max_iter:
            step = project_simplex(a - (a @ gram - cross) / lipschitz)
            residual = float(np.linalg.norm(step - a, axis=1).max())
            if residual <= tolerance:
                converged = True
                break
    return a, {
        "iterations": iteration,
        "converged": converged,
        "projected_step_residual": residual,
        "tolerance": tolerance,
    }


def grid_laplacian(a, shape):
    """Four-neighbor grid gradient with no flattened-row wrap-around."""
    image = a.reshape(*shape, -1)
    output = np.zeros_like(image)
    vertical, horizontal = image[:-1] - image[1:], image[:, :-1] - image[:, 1:]
    output[:-1] += vertical
    output[1:] -= vertical
    output[:, :-1] += horizontal
    output[:, 1:] -= horizontal
    return output.reshape(a.shape)


def spatial_energy(a, shape):
    image = a.reshape(*shape, -1)
    return float(np.sum(np.diff(image, axis=0) ** 2) + np.sum(np.diff(image, axis=1) ** 2))


def constrained_nmf(
    x,
    initial_endmembers,
    shape,
    seed=0,
    spatial_weight=0.0,
    max_iter=300,
    inner_steps=15,
    tolerance=1e-7,
    initialization_jitter=0.02,
    fcls_max_iter=3000,
    fcls_tolerance=1e-8,
):
    """Alternating projected gradients: nonnegative E, simplex A, optional smoothing.

    Loss = ||X-AE||²/(2NB) + spatial_weight*sum_edges ||a_i-a_j||²/(2N).
    """
    x = _validate_spectra(x)
    e = _validate_spectra(initial_endmembers).copy()
    if (
        len(shape) != 2
        or np.prod(shape) != len(x)
        or e.shape[1] != x.shape[1]
        or spatial_weight < 0
        or initialization_jitter < 0
        or max_iter < 1
        or inner_steps < 1
        or tolerance <= 0
    ):
        raise ValueError("Invalid NMF dimensions or settings.")
    rng = np.random.default_rng(seed)
    e = np.maximum(e + initialization_jitter * x.std() * rng.normal(size=e.shape), 1e-10)
    a, initial_solver = fcls(x, e, fcls_max_iter, fcls_tolerance)
    n, bands = x.shape
    x_norm2 = float(np.sum(x * x))
    history, previous, small_changes = [], float("inf"), 0
    converged = False
    for iteration in range(1, max_iter + 1):
        gram_a, cross_a = a.T @ a, a.T @ x
        step_e = max(float(np.linalg.eigvalsh(gram_a)[-1]), 1e-12)
        for _ in range(3):
            e = np.maximum(e - (gram_a @ e - cross_a) / step_e, 0)
        gram_e, cross_e = (e @ e.T) / bands, (x @ e.T) / bands
        step_a = max(float(np.linalg.eigvalsh(gram_e)[-1]) + 8 * spatial_weight, 1e-12)
        for _ in range(inner_steps):
            gradient = a @ gram_e - cross_e
            if spatial_weight:
                gradient += spatial_weight * grid_laplacian(a, shape)
            a = project_simplex(a - gradient / step_a)
        if iteration % 5 == 0 or iteration == max_iter:
            gram_a, cross_a = a.T @ a, a.T @ x
            error2 = max(0.0, x_norm2 - 2 * np.sum(e * cross_a) + np.sum((e @ e.T) * gram_a))
            loss = error2 / (2 * n * bands) + spatial_weight * spatial_energy(a, shape) / (
                2 * n
            )
            history.append(
                {
                    "iteration": iteration,
                    "objective": float(loss),
                    "relative_reconstruction_error": float(np.sqrt(error2 / x_norm2)),
                }
            )
            change = (
                abs(previous - loss) / max(abs(previous), 1e-15)
                if np.isfinite(previous)
                else float("inf")
            )
            small_changes = small_changes + 1 if change < tolerance else 0
            previous = loss
            if small_changes >= 4:
                converged = True
                break
    return UnmixingResult(
        e,
        a,
        {
            "iterations": iteration,
            "converged": converged,
            "stopping_rule": (
                "relative objective change below tolerance for four checks, every 5 steps"
            ),
            "tolerance": tolerance,
            "spatial_weight": spatial_weight,
            "initialization_fcls": initial_solver,
            "history": history,
        },
    )


def unmix(cube, n_components, method="simplex-nmf", seed=0, **options):
    cube = np.asarray(cube, dtype=np.float64)
    if cube.ndim != 3:
        raise ValueError("Expected a rows x columns x bands cube.")
    start = perf_counter()
    x = cube.reshape(-1, cube.shape[-1])
    initial, selected = spa(x, n_components)
    if method == "spa-fcls":
        a, diagnostics = fcls(
            x, initial, options.get("fcls_max_iter", 3000), options.get("fcls_tolerance", 1e-8)
        )
        result = UnmixingResult(initial, a, diagnostics)
    elif method in ("simplex-nmf", "spatial-nmf"):
        keys = {
            "max_iter",
            "inner_steps",
            "tolerance",
            "initialization_jitter",
            "fcls_max_iter",
            "fcls_tolerance",
        }
        arguments = {k: v for k, v in options.items() if k in keys}
        weight = options.get("spatial_weight", 0.002) if method == "spatial-nmf" else 0.0
        result = constrained_nmf(
            x, initial, cube.shape[:2], seed=seed, spatial_weight=weight, **arguments
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    result.diagnostics.update(
        {
            "fit_seconds": perf_counter() - start,
            "method": method,
            "seed": seed,
            "spa_pixel_indices": selected,
        }
    )
    return result
