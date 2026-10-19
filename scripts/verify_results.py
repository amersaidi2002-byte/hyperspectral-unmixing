"""Verify result checksums and recompute representative metrics from saved fits."""

import argparse
from pathlib import Path

import numpy as np

from spectral_unmixing.algorithms import UnmixingResult
from spectral_unmixing.artifacts import load_saved, scene_for_run
from spectral_unmixing.experiment import METRICS, load_runs
from spectral_unmixing.metrics import evaluate


def verify(results, data):
    record, runs = load_runs(results)
    count = 0
    for run, relative in zip(runs, record["runs"], strict=True):
        assert run["metrics"]["min_abundance"] >= 0
        assert run["metrics"]["max_sum_to_one_error"] < 1e-12
        if "array_sha256" not in run:
            continue
        scene = scene_for_run(run, record["config"], data)
        _, e, a = load_saved(results, relative)
        result = UnmixingResult(e, a.reshape(-1, len(e)), run["diagnostics"])
        measured, _, _ = evaluate(scene, result)
        for metric in METRICS:
            # arccos loses precision near a zero angle; reordering dot products can
            # change their last bits. Two microdegrees is below practical precision.
            tolerance = 2e-6 if metric.endswith(("sad_deg", "sam_deg")) else 1e-9
            np.testing.assert_allclose(
                measured[metric],
                run["metrics"][metric],
                rtol=1e-9,
                atol=tolerance,
                err_msg=f"{relative}: {metric}",
            )
        count += 1
    print(f"Verified {len(runs)} records and recomputed metrics for {count} saved fits.")
    return len(runs), count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    verify(args.results, args.data)
