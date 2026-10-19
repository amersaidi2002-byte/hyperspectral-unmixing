import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from spectral_unmixing import data
from spectral_unmixing.algorithms import UnmixingResult
from spectral_unmixing.metrics import evaluate, spectral_angles


def test_exact_mixtures_and_no_pure_bound():
    scene = data.synthetic_scene(size=12, bands=24, snr_db=None, regime="mixed")
    np.testing.assert_allclose(
        scene.spectra, scene.abundances.reshape(-1, 3) @ scene.endmembers
    )
    assert scene.abundances.min() >= 0.1 and scene.abundances.max() <= 0.8


def test_noise_and_determinism():
    a = data.synthetic_scene(size=12, bands=24, seed=14, snr_db=20)
    b = data.synthetic_scene(size=12, bands=24, seed=14, snr_db=20)
    np.testing.assert_array_equal(a.cube, b.cube)
    assert a.metadata["actual_snr_db"] == pytest.approx(20, abs=0.05)


def test_permutation_alignment():
    scene = data.synthetic_scene(size=12, bands=24, snr_db=None)
    perm = [2, 0, 1]
    result = UnmixingResult(
        scene.endmembers[perm], scene.abundances.reshape(-1, 3)[:, perm], {"fit_seconds": 0}
    )
    metrics, e, a = evaluate(scene, result)
    assert metrics["abundance_rmse"] < 1e-14
    assert metrics["mean_endmember_sad_deg"] < 2e-6
    np.testing.assert_array_equal(e, scene.endmembers)
    np.testing.assert_array_equal(a, scene.abundances.reshape(-1, 3))


def test_scale_invariant_angles_and_zero_rejection():
    a = np.array([[1.0, 2.0, 3.0]])
    assert spectral_angles(a, 7 * a)[0, 0] < 2e-6
    with pytest.raises(ValueError, match="zero"):
        spectral_angles(a, np.zeros_like(a))


def test_bundled_source_manifest():
    assert data.source_manifest() == json.loads(Path("configs/sources.json").read_text())
    assert len(data.source_manifest()["files"]) == 4


def test_modified_cache(tmp_path, monkeypatch):
    item = {"filename": "bad.mat", "sha256": "0" * 64, "url": "https://example.invalid"}
    monkeypatch.setattr(data, "source_manifest", lambda: {"files": [item]})
    (tmp_path / "bad.mat").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.download_data(tmp_path)


def test_matlab_pixel_order(tmp_path, monkeypatch):
    h, w, b, k = 95, 95, 156, 3
    rng = np.random.default_rng(3)
    a, e = rng.dirichlet([1, 1, 1], size=h * w), rng.uniform(0.1, 1, (k, b))
    x = a @ e
    raw, gt = {"V": x.T}, {"A": a.T, "M": e.T}
    items = []
    for filename in ("Samson.mat", "Samson_GT.mat"):
        (tmp_path / filename).write_bytes(b"fixture")
        items.append({"filename": filename, "sha256": hashlib.sha256(b"fixture").hexdigest()})
    monkeypatch.setattr(
        data, "source_manifest", lambda: {"files": items, "revision": "fixture"}
    )
    monkeypatch.setattr(data, "loadmat", lambda p: gt if "GT" in p.name else raw)
    scene = data.load_scene("samson", tmp_path)
    np.testing.assert_allclose(scene.cube[7, 9], x[7 + 9 * h] / x.max())
    np.testing.assert_allclose(scene.abundances[7, 9], a[7 + 9 * h])
    np.testing.assert_allclose(scene.cube, scene.abundances @ scene.endmembers)
