"""Fixed-budget per-scene experiments; reference arrays are used only after fitting."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .algorithms import unmix
from .data import load_scene, sha256, source_manifest, synthetic_scene
from .metrics import evaluate

METHODS = ("spa-fcls", "simplex-nmf", "spatial-nmf")
METRICS = (
    "mean_endmember_sad_deg",
    "abundance_rmse",
    "relative_reconstruction_error",
    "mean_reconstruction_sam_deg",
    "fit_seconds",
)
SOLVER_KEYS = (
    "max_iter",
    "inner_steps",
    "tolerance",
    "spatial_weight",
    "initialization_jitter",
    "fcls_max_iter",
    "fcls_tolerance",
)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text())
    required = {
        "real_scenes",
        "seeds",
        "methods",
        *SOLVER_KEYS,
        "synthetic_size",
        "synthetic_bands",
        "synthetic_snr_db",
        "synthetic_regimes",
        "threads",
    }
    if set(config) != required:
        raise ValueError(f"Configuration keys differ: {set(config) ^ required}")
    if (
        not config["seeds"]
        or len(config["seeds"]) != len(set(config["seeds"]))
        or 0 not in config["seeds"]
        or any(not isinstance(s, int) or s < 0 for s in config["seeds"])
        or not config["methods"]
        or not set(config["methods"]) <= set(METHODS)
        or not set(config["real_scenes"]) <= {"samson", "jasper"}
        or not set(config["synthetic_regimes"]) <= {"pure", "mixed"}
        or config["threads"] < 1
    ):
        raise ValueError("Invalid scenes, methods, seeds, regimes or threads.")
    return config


def environment():
    packages = {
        n: importlib.metadata.version(n)
        for n in ("numpy", "scipy", "matplotlib", "threadpoolctl", "hyperspectral-unmixing")
    }
    cpu = platform.processor()
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": cpu,
        "packages": packages,
    }


def run_benchmark(config, data_root, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty: {output}. Choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "created_utc": datetime.now(UTC).isoformat(),
        "config": config,
        "config_sha256": config_digest(config),
        "environment": environment(),
        "sources": source_manifest(),
        "runs": [],
    }
    write_json(output / "experiment.json", record)
    settings = {k: config[k] for k in SOLVER_KEYS}

    def run(scene, case, method, seed, save_arrays):
        print(f"{scene.name:10} {case:14} {method:12} seed={seed}", flush=True)
        with threadpool_limits(limits=config["threads"]):
            result = unmix(scene.cube, len(scene.materials), method, seed, **settings)
            metrics, e, a = evaluate(scene, result)
        target = output / scene.name / case / method / f"seed-{seed}"
        report = {
            "scene": scene.name,
            "case": case,
            "method": method,
            "seed": seed,
            "scene_metadata": scene.metadata,
            "config_sha256": record["config_sha256"],
            "metrics": metrics,
            "diagnostics": result.diagnostics,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        if save_arrays:
            np.savez_compressed(
                target.with_suffix(".npz"),
                endmembers=e,
                abundances=a.reshape(*scene.cube.shape[:2], -1),
            )
            report["array_sha256"] = sha256(target.with_suffix(".npz"))
        write_json(target.with_suffix(".json"), report)
        record["runs"].append(str(target.relative_to(output).with_suffix(".json")))
        write_json(output / "experiment.json", record)

    for name in config["real_scenes"]:
        scene = load_scene(name, data_root)
        for method in config["methods"]:
            for seed in [0] if method == "spa-fcls" else config["seeds"]:
                run(scene, "original", method, seed, seed == 0)
    for regime in config["synthetic_regimes"]:
        for snr in config["synthetic_snr_db"]:
            case = f"{regime}-{'clean' if snr is None else f'snr{snr}'}"
            for seed in config["seeds"]:
                scene = synthetic_scene(
                    config["synthetic_size"], config["synthetic_bands"], seed, snr, regime
                )
                for method in config["methods"]:
                    run(scene, case, method, seed, seed == 0)
    record["completed_utc"] = datetime.now(UTC).isoformat()
    write_json(output / "experiment.json", record)
    summarize(output)
    return record


def load_runs(output):
    output = Path(output)
    record = json.loads((output / "experiment.json").read_text())
    if "completed_utc" not in record:
        raise ValueError("The experiment did not complete; refusing a partial summary.")
    if len(record["runs"]) != len(set(record["runs"])):
        raise ValueError("Duplicate run records.")
    runs = [json.loads((output / p).read_text()) for p in record["runs"]]
    for run in runs:
        if run["config_sha256"] != config_digest(record["config"]):
            raise ValueError("Run configuration differs from experiment.")
    return record, runs


def summarize(output):
    _, runs = load_runs(output)
    groups = defaultdict(list)
    for run in runs:
        groups[(run["scene"], run["case"], run["method"])].append(run)
    rows = []
    for (scene, case, method), group in sorted(groups.items()):
        row = {"scene": scene, "case": case, "method": method, "n_runs": len(group)}
        for metric in METRICS:
            values = [r["metrics"][metric] for r in group]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        rows.append(row)
    if not rows:
        raise ValueError("No runs to summarize.")
    with (Path(output) / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(Path(output) / "summary.json", rows)
    return rows
