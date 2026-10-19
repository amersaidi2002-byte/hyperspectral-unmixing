import json
from pathlib import Path
from threading import Thread
from urllib.error import HTTPError
from urllib.request import ProxyHandler, build_opener

import numpy as np
import pytest

from spectral_unmixing.artifacts import load_saved
from spectral_unmixing.cli import main
from spectral_unmixing.data import synthetic_scene
from spectral_unmixing.experiment import load_runs, read_config, run_benchmark
from spectral_unmixing.explorer import Explorer, create_server


@pytest.fixture
def small_experiment(tmp_path):
    config = read_config(Path("configs/benchmark.json"))
    config.update(
        real_scenes=[],
        seeds=[0],
        max_iter=10,
        synthetic_size=12,
        synthetic_bands=16,
        synthetic_snr_db=[35],
        synthetic_regimes=["pure"],
    )
    output = tmp_path / "experiment"
    run_benchmark(config, tmp_path / "missing", output)
    return output


def test_offline_workflow_and_http(small_experiment, tmp_path):
    server = create_server(Explorer(small_experiment, tmp_path / "missing"))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with opener.open(base + "/api/scenes") as response:
            scene = json.load(response)["scenes"][0]
        assert scene["id"] == "synthetic" and len(scene["methods"]) == 3
        for method in scene["methods"]:
            query = f"scene=synthetic&method={method}&row=4&column=7"
            with opener.open(base + "/api/pixel?" + query) as response:
                pixel = json.load(response)
            assert sum(pixel["estimated"]) == pytest.approx(1, abs=1e-12)
            np.testing.assert_allclose(
                np.sum(pixel["contributions"], axis=0), pixel["reconstructed"]
            )
            assert len(pixel["observed"]) == 16
            with opener.open(base + "/api/image?" + query + "&component=0&view=estimate") as r:
                assert r.headers["Content-Type"] == "image/png"
                assert r.read().startswith(b"\x89PNG\r\n\x1a\n")
        with pytest.raises(HTTPError) as error:
            opener.open(base + "/api/pixel?scene=synthetic&method=spa-fcls&row=-1&column=0")
        assert error.value.code == 400
        with pytest.raises(HTTPError) as error:
            opener.open(base + "/../configs/benchmark.json")
        assert error.value.code == 404
        with opener.open(base + "/") as response:
            assert b"What is inside a pixel?" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_modified_array_rejected(small_experiment):
    record, _ = load_runs(small_experiment)
    path = small_experiment / record["runs"][0]
    path.with_suffix(".npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_saved(small_experiment, record["runs"][0])


def test_partial_experiment_rejected(tmp_path):
    (tmp_path / "experiment.json").write_text(json.dumps({"runs": []}))
    with pytest.raises(ValueError, match="did not complete"):
        load_runs(tmp_path)


def test_cli_custom_cube_without_labels(tmp_path):
    np.save(tmp_path / "cube.npy", synthetic_scene(size=12, bands=16).cube)
    main(
        [
            "fit",
            "--input",
            str(tmp_path / "cube.npy"),
            "--components",
            "3",
            "--method",
            "spa-fcls",
            "--output",
            str(tmp_path / "fit.npz"),
        ]
    )
    with np.load(tmp_path / "fit.npz", allow_pickle=False) as arrays:
        assert arrays["endmembers"].shape == (3, 16)
        assert arrays["abundances"].shape == (12, 12, 3)
    assert json.loads((tmp_path / "fit.json").read_text())["converged"]
