"""Pinned real-scene loaders and an analytic, spatial linear-mixture generator."""

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


@dataclass
class Scene:
    name: str
    cube: np.ndarray
    endmembers: np.ndarray
    abundances: np.ndarray
    materials: tuple[str, ...]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        h, w, b = self.cube.shape
        k = len(self.materials)
        if self.endmembers.shape != (k, b) or self.abundances.shape != (h, w, k):
            raise ValueError("Cube, reference spectra and abundance dimensions disagree.")
        for a in (self.cube, self.endmembers, self.abundances):
            if not np.isfinite(a).all() or a.min() < -1e-8:
                raise ValueError("Scene arrays must be finite and nonnegative.")
        if not np.allclose(self.abundances.sum(axis=-1), 1, atol=1e-5):
            raise ValueError("Reference fractions must sum to one.")

    @property
    def spectra(self):
        return self.cube.reshape(-1, self.cube.shape[-1])


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_manifest():
    return json.loads(files("spectral_unmixing.resources").joinpath("sources.json").read_text())


def download_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest()
    for item in manifest["files"]:
        destination = root / item["filename"]
        if destination.exists():
            if sha256(destination) != item["sha256"]:
                raise ValueError(
                    f"Checksum mismatch: {destination}. Remove it and download again."
                )
            continue
        temporary = destination.with_suffix(".part")
        try:
            request = urllib.request.Request(
                item["url"], headers={"User-Agent": "hsi-unmix/0.1"}
            )
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                temporary.open("wb") as out,
            ):
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if sha256(temporary) != item["sha256"]:
                raise ValueError(
                    f"Downloaded file failed SHA-256 verification: {destination.name}"
                )
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def load_scene(name, root):
    specs = {
        "samson": (
            "Samson.mat",
            "Samson_GT.mat",
            "V",
            (95, 95, 156),
            ("Rock/soil", "Tree", "Water"),
        ),
        "jasper": (
            "jasperRidge2_R198.mat",
            "Jasper_GT.mat",
            "Y",
            (100, 100, 198),
            ("Tree", "Water", "Dirt", "Road"),
        ),
    }
    if name not in specs:
        raise ValueError(f"Unknown scene: {name}")
    cube_file, gt_file, variable, shape, materials = specs[name]
    manifest = source_manifest()
    expected = {r["filename"]: r for r in manifest["files"]}
    for filename in (cube_file, gt_file):
        path = Path(root) / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Run hsi-unmix download --data {root}")
        if sha256(path) != expected[filename]["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path}")
    raw, reference = loadmat(Path(root) / cube_file), loadmat(Path(root) / gt_file)
    h, w, bands = shape
    spectra = np.asarray(raw[variable], dtype=np.float64)
    if spectra.shape != (bands, h * w):
        raise ValueError(f"Unexpected source dimensions: {spectra.shape}")
    endmembers = np.asarray(reference["M"], dtype=np.float64).T
    divisor = float(np.asarray(raw.get("maxValue", [[1.0]])).item())
    spectra /= divisor
    scale = float(spectra.max())
    spectra /= scale
    # Samson M uses separate component amplitude normalization. Only scale-invariant
    # spectral angles compare references; reference E is not a reflectance oracle.
    endmembers /= scale
    cube = spectra.T.reshape(h, w, bands, order="F")
    abundances = reference["A"].T.reshape(h, w, -1, order="F")
    return Scene(
        name,
        cube,
        endmembers,
        abundances,
        materials,
        {
            "source_revision": manifest["revision"],
            "source_sha256": {f: expected[f]["sha256"] for f in (cube_file, gt_file)},
            "raw_divisor": divisor,
            "global_scale": scale,
            "spectral_axis": "retained band index (wavelength calibration not supplied)",
            "reference_status": "benchmark reference estimates, not field-measured fractions",
            "material_labels": "source cood entries; Samson 1-rock displayed as Rock/soil",
            "reference_spectral_comparison": "scale-invariant SAD; amplitudes not calibrated",
        },
    )


def synthetic_scene(size=48, bands=96, seed=0, snr_db=35.0, regime="pure"):
    """Analytic spectra and spatial mixtures, not measured mineral signatures."""
    if size < 12 or bands < 16 or regime not in ("pure", "mixed"):
        raise ValueError("Use size >= 12, bands >= 16, and regime pure or mixed.")
    if snr_db is not None and (not np.isfinite(snr_db) or snr_db <= 0):
        raise ValueError("snr_db must be positive and finite, or None.")
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, bands)
    e = np.array(
        [
            0.30 + 0.37 * t - 0.12 * np.exp(-(((t - 0.64) / 0.09) ** 2)),
            0.16
            + 0.48 / (1 + np.exp(-22 * (t - 0.44)))
            - 0.16 * np.exp(-(((t - 0.83) / 0.08) ** 2)),
            0.12 + 0.20 * np.exp(-4 * t) + 0.035 * np.sin(9 * t) ** 2,
        ]
    )
    yy, xx = np.mgrid[0 : 1 : complex(size), 0 : 1 : complex(size)]
    weights = np.stack(
        [
            np.exp(-((xx - 0.15) ** 2 + (yy - 0.25) ** 2) / 0.12),
            np.exp(-((xx - 0.8) ** 2 + (yy - 0.25) ** 2) / 0.13),
            np.exp(-((xx - 0.5) ** 2 + (yy - 0.85) ** 2) / 0.13),
        ],
        axis=-1,
    )
    texture = gaussian_filter(rng.normal(size=(size, size)), 2)
    weights *= np.exp(0.15 * texture[..., None] * np.array([1, -1, 0.5]))
    a = weights / weights.sum(axis=-1, keepdims=True)
    if regime == "pure":
        s = max(2, size // 10)
        for j, (r, c) in enumerate(((s, s), (s, size - 2 * s), (size - 2 * s, size // 2))):
            a[r : r + s, c : c + s] = np.eye(3)[j]
    else:
        a = 0.7 * a + 0.1
    clean = a @ e
    if snr_db is None:
        noisy, actual_snr, clipped = clean.copy(), None, 0
    else:
        noise = rng.normal(size=clean.shape)
        noise *= np.linalg.norm(clean) / (10 ** (snr_db / 20) * np.linalg.norm(noise))
        noisy = clean + noise
        clipped = int(np.count_nonzero(noisy < 0))
        noisy = np.maximum(noisy, 0)
        actual_snr = float(20 * np.log10(np.linalg.norm(clean) / np.linalg.norm(noisy - clean)))
    return Scene(
        "synthetic",
        noisy,
        e,
        a,
        ("Soil-like", "Vegetation-like", "Water-like"),
        {
            "synthetic": True,
            "seed": seed,
            "regime": regime,
            "requested_snr_db": snr_db,
            "actual_snr_db": actual_snr,
            "clipped_values": clipped,
            "spectral_axis": "synthetic band index",
            "reference_status": "exact analytic linear-mixture ground truth",
        },
    )


def false_color(cube):
    indices = np.round(np.array([0.7, 0.45, 0.15]) * (cube.shape[-1] - 1)).astype(int)
    image = cube[..., indices].copy()
    lo, hi = np.percentile(image, [2, 98], axis=(0, 1))
    return np.clip((image - lo) / np.maximum(hi - lo, 1e-12), 0, 1)
