"""Localhost-only pixel inspector for checksum-verified saved fits."""

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlsplit

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import numpy as np

from .artifacts import load_saved, scene_for_run
from .data import false_color
from .experiment import load_runs


class Explorer:
    def __init__(self, results, data_root):
        record, runs = load_runs(results)
        self.scenes, self.fits, self.notes = {}, {}, []
        for run, relative in zip(runs, record["runs"], strict=True):
            if run["seed"] != 0:
                continue
            if run["scene"] == "synthetic" and run["case"] != "pure-snr35":
                continue
            name = run["scene"]
            if name not in self.scenes:
                try:
                    self.scenes[name] = scene_for_run(run, record["config"], data_root)
                except FileNotFoundError:
                    note = f"{name}: run hsi-unmix download to inspect this real scene."
                    if note not in self.notes:
                        self.notes.append(note)
                    continue
            _, e, a = load_saved(results, relative)
            scene = self.scenes[name]
            if e.shape != scene.endmembers.shape or a.shape != scene.abundances.shape:
                raise ValueError("Saved fit dimensions differ from source scene.")
            self.fits[(name, run["method"])] = (e, a, run)
        if not self.scenes:
            raise ValueError(
                "No scene available. Include synthetic pure-snr35 or download data."
            )

    def manifest(self):
        return {
            "scenes": [
                {
                    "id": name,
                    "height": scene.cube.shape[0],
                    "width": scene.cube.shape[1],
                    "bands": scene.cube.shape[2],
                    "materials": scene.materials,
                    "metadata": scene.metadata,
                    "methods": [m for n, m in self.fits if n == name],
                }
                for name, scene in self.scenes.items()
            ],
            "notes": self.notes,
        }

    def pixel(self, name, method, row, column):
        scene = self.scenes[name]
        e, a, run = self.fits[(name, method)]
        if not (0 <= row < scene.cube.shape[0] and 0 <= column < scene.cube.shape[1]):
            raise ValueError("Pixel coordinates are outside the image.")
        observed, fractions = scene.cube[row, column], a[row, column]
        reconstruction = fractions @ e
        return {
            "row": row,
            "column": column,
            "observed": observed.tolist(),
            "reconstructed": reconstruction.tolist(),
            "contributions": (fractions[:, None] * e).tolist(),
            "estimated": fractions.tolist(),
            "reference": scene.abundances[row, column].tolist(),
            "spectral_rmse": float(np.sqrt(np.mean((observed - reconstruction) ** 2))),
            "metrics": run["metrics"],
        }

    def image(self, name, method, view, component):
        scene = self.scenes[name]
        if view == "rgb":
            array = false_color(scene.cube)
        else:
            if not 0 <= component < len(scene.materials):
                raise ValueError("Invalid component.")
            if view == "reference":
                array = scene.abundances[..., component]
            elif view == "estimate":
                array = self.fits[(name, method)][1][..., component]
            else:
                raise ValueError("Invalid view.")
        stream = io.BytesIO()
        mpimg.imsave(stream, array, format="png", vmin=0, vmax=1, cmap="viridis")
        return stream.getvalue()


def create_server(explorer, port=0):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            query = {k: v[0] for k, v in parse_qs(url.query).items()}
            try:
                if url.path in ("/", "/app.js", "/style.css"):
                    filename, mime = {
                        "/": ("index.html", "text/html; charset=utf-8"),
                        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                        "/style.css": ("style.css", "text/css; charset=utf-8"),
                    }[url.path]
                    body = files("spectral_unmixing").joinpath("web", filename).read_bytes()
                elif url.path == "/api/scenes":
                    body, mime = json.dumps(explorer.manifest()).encode(), "application/json"
                elif url.path == "/api/pixel":
                    pixel = explorer.pixel(
                        query["scene"], query["method"], int(query["row"]), int(query["column"])
                    )
                    body = json.dumps(pixel, allow_nan=False).encode()
                    mime = "application/json"
                elif url.path == "/api/image":
                    body = explorer.image(
                        query["scene"],
                        query["method"],
                        query["view"],
                        int(query.get("component", 0)),
                    )
                    mime = "image/png"
                else:
                    self.send(b"Not found", "text/plain", 404)
                    return
                self.send(body, mime)
            except (ValueError, KeyError) as error:
                self.send(json.dumps({"error": str(error)}).encode(), "application/json", 400)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def serve(results, data_root, port=7861):
    server = create_server(Explorer(results, data_root), port)
    print(f"Pixel inspector: http://127.0.0.1:{server.server_address[1]}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
