"""Reference comparisons are separate from unsupervised fitting."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def spectral_angles(reference, estimate):
    reference, estimate = np.asarray(reference), np.asarray(estimate)
    if reference.ndim != 2 or estimate.ndim != 2 or reference.shape[1] != estimate.shape[1]:
        raise ValueError("Spectral matrices must share their band dimension.")
    nr, ne = np.linalg.norm(reference, axis=1), np.linalg.norm(estimate, axis=1)
    if np.any(nr == 0) or np.any(ne == 0):
        raise ValueError("Spectral angles are undefined for zero spectra.")
    cosine = (reference @ estimate.T) / (nr[:, None] * ne[None, :])
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def evaluate(scene, result):
    e, a = result.endmembers, result.abundances
    cost = spectral_angles(scene.endmembers, e)
    rows, permutation = linear_sum_assignment(cost)
    if len(rows) != len(scene.materials) or len(e) != len(scene.materials):
        raise ValueError("Reference and estimated component counts must agree.")
    aligned_e, aligned_a = e[permutation], a[:, permutation]
    reference_a = scene.abundances.reshape(aligned_a.shape)
    reconstruction, x = a @ e, scene.spectra
    norms = np.linalg.norm(x, axis=1) * np.linalg.norm(reconstruction, axis=1)
    valid = norms > 0
    angles = np.degrees(
        np.arccos(
            np.clip(np.sum(x[valid] * reconstruction[valid], axis=1) / norms[valid], -1, 1)
        )
    )
    return (
        {
            "mean_endmember_sad_deg": float(cost[rows, permutation].mean()),
            "abundance_rmse": float(np.sqrt(np.mean((aligned_a - reference_a) ** 2))),
            "relative_reconstruction_error": float(
                np.linalg.norm(x - reconstruction) / np.linalg.norm(x)
            ),
            "mean_reconstruction_sam_deg": float(angles.mean()),
            "reconstruction_sam_valid_pixels": int(valid.sum()),
            "per_component_sad_deg": cost[rows, permutation].tolist(),
            "per_component_abundance_rmse": np.sqrt(
                np.mean((aligned_a - reference_a) ** 2, axis=0)
            ).tolist(),
            "reference_to_estimate_permutation": permutation.tolist(),
            "min_abundance": float(a.min()),
            "max_sum_to_one_error": float(np.max(abs(a.sum(axis=1) - 1))),
            "fit_seconds": result.diagnostics["fit_seconds"],
        },
        aligned_e,
        aligned_a,
    )
