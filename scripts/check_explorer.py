"""Exercise the saved scenes and methods through the actual localhost HTTP API."""

import argparse
import json
from pathlib import Path
from threading import Thread
from urllib.request import ProxyHandler, build_opener

import numpy as np

from spectral_unmixing.explorer import Explorer, create_server


def check(results, data):
    explorer = Explorer(results, data)
    server = create_server(explorer)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    opener = build_opener(ProxyHandler({}))
    base = f"http://127.0.0.1:{server.server_address[1]}"
    count = 0
    try:
        for scene in explorer.manifest()["scenes"]:
            for method in scene["methods"]:
                for row, col in [
                    (0, 0),
                    (scene["height"] // 2, scene["width"] // 2),
                    (scene["height"] - 1, scene["width"] - 1),
                ]:
                    query = f"scene={scene['id']}&method={method}&row={row}&column={col}"
                    with opener.open(base + "/api/pixel?" + query) as response:
                        pixel = json.load(response)
                    np.testing.assert_allclose(sum(pixel["estimated"]), 1, atol=1e-12)
                    np.testing.assert_allclose(
                        np.sum(pixel["contributions"], axis=0),
                        pixel["reconstructed"],
                        atol=1e-12,
                    )
                    for view in ["rgb", "reference", "estimate"]:
                        with opener.open(
                            base + "/api/image?" + query + f"&view={view}&component=0"
                        ) as r:
                            assert r.read().startswith(b"\x89PNG\r\n\x1a\n")
                    count += 1
        print(f"Passed {count} pixel checks and {3 * count} PNG responses over HTTP.")
        return count
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    check(args.results, args.data)
