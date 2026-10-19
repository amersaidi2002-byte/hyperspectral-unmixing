"""Load saved fits with checksum verification and recreate their input scenes."""

import json
from pathlib import Path

import numpy as np

from .data import load_scene, sha256, synthetic_scene


def load_saved(root, relative_run):
    root = Path(root).resolve()
    path = (root / relative_run).resolve()
    if root not in path.parents or path.suffix != ".json":
        raise ValueError("Invalid run path.")
    report = json.loads(path.read_text())
    array_path = path.with_suffix(".npz")
    if sha256(array_path) != report.get("array_sha256"):
        raise ValueError(f"Saved result checksum mismatch: {array_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        e, a = arrays["endmembers"], arrays["abundances"]
    if not np.isfinite(e).all() or not np.isfinite(a).all():
        raise ValueError("Non-finite saved arrays.")
    return report, e, a


def scene_for_run(run, config, data_root):
    if run["scene"] != "synthetic":
        return load_scene(run["scene"], data_root)
    meta = run["scene_metadata"]
    return synthetic_scene(
        config["synthetic_size"],
        config["synthetic_bands"],
        run["seed"],
        meta["requested_snr_db"],
        meta["regime"],
    )
