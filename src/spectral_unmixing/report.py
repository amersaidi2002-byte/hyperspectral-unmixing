"""Scientific figures and Markdown generated from complete measured experiments."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .artifacts import load_saved, scene_for_run
from .data import false_color
from .experiment import METHODS, load_runs, summarize

LABELS = {"spa-fcls": "SPA + FCLS", "simplex-nmf": "Simplex NMF", "spatial-nmf": "Spatial NMF"}
COLORS = {"spa-fcls": "#475569", "simplex-nmf": "#2563eb", "spatial-nmf": "#0d9488"}


def save(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def representatives(results, record, data_root):
    selections = {}
    for relative in record["runs"]:
        if Path(relative).name != "seed-0.json":
            continue
        run, e, a = load_saved(results, relative)
        key = (run["scene"], run["case"])
        if key not in selections:
            selections[key] = {
                "scene": scene_for_run(run, record["config"], data_root),
                "fits": {},
            }
        selections[key]["fits"][run["method"]] = (e, a, run)
    return selections


def abundance_figure(selection, target):
    scene, fits = selection["scene"], selection["fits"]
    methods = [m for m in METHODS if m in fits]
    fig, axes = plt.subplots(
        len(scene.materials),
        len(methods) + 1,
        figsize=(12, 2.35 * len(scene.materials)),
        squeeze=False,
        layout="constrained",
    )
    for j, material in enumerate(scene.materials):
        image = axes[j, 0].imshow(scene.abundances[..., j], vmin=0, vmax=1, cmap="viridis")
        axes[j, 0].set_ylabel(material)
        if j == 0:
            axes[j, 0].set_title(
                "Exact reference" if scene.name == "synthetic" else "Reference estimate"
            )
        for col, method in enumerate(methods, 1):
            axes[j, col].imshow(fits[method][1][..., j], vmin=0, vmax=1, cmap="viridis")
            if j == 0:
                axes[j, col].set_title(LABELS[method])
        for ax in axes[j]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.65, label="Material fraction · shared [0, 1] scale")
    fig.suptitle(f"{scene.name.title()} · seed 0 · reference alignment only after fitting")
    save(fig, target)


def overview_figure(selection, target):
    scene = selection["scene"]
    method = (
        "simplex-nmf" if "simplex-nmf" in selection["fits"] else next(iter(selection["fits"]))
    )
    e, a, _ = selection["fits"][method]
    k = len(scene.materials)
    fig = plt.figure(figsize=(13, 7), layout="constrained")
    grid = fig.add_gridspec(2, k + 1)
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(false_color(scene.cube))
    ax.set_title("False-color input")
    ax.axis("off")
    image_axes = [ax]
    for j, material in enumerate(scene.materials):
        ax = fig.add_subplot(grid[0, j + 1])
        image = ax.imshow(a[..., j], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(material)
        ax.axis("off")
        image_axes.append(ax)
    fig.colorbar(image, ax=image_axes, shrink=0.7, fraction=0.02, label="Estimated fraction")
    ax = fig.add_subplot(grid[1, :])
    r, c = scene.cube.shape[0] // 2, scene.cube.shape[1] // 2
    ax.plot(scene.cube[r, c], color="#0f172a", lw=2, label="Observed pixel")
    ax.plot(a[r, c] @ e, "--", color="#dc2626", lw=1.8, label="Reconstruction")
    for j, material in enumerate(scene.materials):
        ax.plot(a[r, c, j] * e[j], lw=1, label=material + " contribution")
    ax.set(xlabel="Retained band index", ylabel="Globally scaled signal")
    ax.grid(alpha=0.12)
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="outside lower center",
        ncol=k + 2,
        fontsize=8,
        frameon=False,
    )
    fig.suptitle(
        f"Hyperspectral unmixing · {scene.name.title()} · {LABELS[method]}\n"
        "Estimate material fractions and inspect spectral residuals",
        fontsize=15,
    )
    save(fig, target)


def benchmark_figure(summary, target):
    scenes = sorted({r["scene"] for r in summary if r["scene"] != "synthetic"})
    if not scenes:
        return
    fig, axes = plt.subplots(
        len(scenes), 3, figsize=(12, 3.2 * len(scenes)), squeeze=False, layout="constrained"
    )
    metrics = [
        ("mean_endmember_sad_deg", "Endmember angle (degrees)"),
        ("abundance_rmse", "Abundance RMSE"),
        ("relative_reconstruction_error", "Relative reconstruction error"),
    ]
    for row, scene in enumerate(scenes):
        lookup = {r["method"]: r for r in summary if r["scene"] == scene}
        methods = [m for m in METHODS if m in lookup]
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            ax.bar(
                range(len(methods)),
                [lookup[m][metric + "_mean"] for m in methods],
                yerr=[lookup[m][metric + "_std"] or 0 for m in methods],
                capsize=4,
                color=[COLORS[m] for m in methods],
                width=0.65,
            )
            ax.set_xticks(
                range(len(methods)), [LABELS[m].replace(" ", "\n", 1) for m in methods]
            )
            ax.set_title(f"{scene.title()}\n{label}")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
    fig.suptitle(
        "Lower is better · mean ± sample SD across NMF initializations\n"
        "SPA: one deterministic run; real-scene references are estimates"
    )
    save(fig, target)


def spectra_figure(selections, target):
    real = [v for (name, _), v in selections.items() if name != "synthetic"]
    if not real:
        return
    fig, axes = plt.subplots(
        len(real), 4, figsize=(13, 3.1 * len(real)), squeeze=False, layout="constrained"
    )
    for row, selection in enumerate(real):
        scene = selection["scene"]
        for j, material in enumerate(scene.materials):
            ax = axes[row, j]
            ref = scene.endmembers[j]
            ax.plot(ref / max(ref.max(), 1e-12), "k--", label="Reference shape", lw=1.8)
            for method, (e, _, _) in selection["fits"].items():
                ax.plot(
                    e[j] / max(e[j].max(), 1e-12),
                    label=LABELS[method],
                    color=COLORS[method],
                    lw=1.2,
                )
            ax.set(
                title=f"{scene.name.title()} · {material}",
                xlabel="Retained band index",
                ylim=(0, 1.05),
            )
            if j == 0:
                ax.set_ylabel("Peak-normalized shape")
        for j in range(len(scene.materials), 4):
            axes[row, j].set_visible(False)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=4,
        frameon=False,
    )
    fig.suptitle("Spectral shape comparison · seed 0\nPeak normalization is for display only")
    save(fig, target)


def synthetic_figure(summary, target):
    rows = [r for r in summary if r["scene"] == "synthetic"]
    if not rows:
        return
    levels = {r["case"].split("-", 1)[1] for r in rows}
    levels = (["clean"] if "clean" in levels else []) + sorted(
        (v for v in levels if v != "clean"), key=lambda s: float(s[3:]), reverse=True
    )
    regimes = [v for v in ["pure", "mixed"] if any(r["case"].startswith(v + "-") for r in rows)]
    fig, axes = plt.subplots(
        len(regimes), 2, figsize=(11, 3.4 * len(regimes)), squeeze=False, layout="constrained"
    )
    for row, regime in enumerate(regimes):
        for col, (metric, label) in enumerate(
            [
                ("mean_endmember_sad_deg", "Endmember angle (degrees)"),
                ("abundance_rmse", "Abundance RMSE"),
            ]
        ):
            ax = axes[row, col]
            for method in METHODS:
                lookup = {
                    r["case"].split("-", 1)[1]: r
                    for r in rows
                    if r["method"] == method and r["case"].startswith(regime + "-")
                }
                if not all(v in lookup for v in levels):
                    continue
                ax.errorbar(
                    range(len(levels)),
                    [lookup[v][metric + "_mean"] for v in levels],
                    yerr=[lookup[v][metric + "_std"] or 0 for v in levels],
                    marker="o",
                    capsize=4,
                    color=COLORS[method],
                    label=LABELS[method],
                )
            ax.set_title(
                f"{'Pure pixels available' if regime == 'pure' else 'No pure pixels'}\n{label}"
            )
            ax.set_xticks(
                range(len(levels)),
                ["Noise-free" if v == "clean" else v[3:] + " dB" for v in levels],
            )
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.15)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Analytic mixtures · mean ± sample SD across scene/noise seeds\n"
        "Mixed regime: fractions in [0.1, 0.8]; exact references"
    )
    save(fig, target)


def error_figure(selections, target):
    real = [v for (name, _), v in selections.items() if name != "synthetic"]
    if not real:
        return
    fig, axes = plt.subplots(
        len(real), 4, figsize=(12, 3 * len(real)), squeeze=False, layout="constrained"
    )
    for row, selection in enumerate(real):
        scene = selection["scene"]
        axes[row, 0].imshow(false_color(scene.cube))
        axes[row, 0].set_title(scene.name.title())
        for col, method in enumerate(METHODS, 1):
            if method not in selection["fits"]:
                axes[row, col].set_visible(False)
                continue
            error = np.sqrt(
                np.mean((selection["fits"][method][1] - scene.abundances) ** 2, axis=-1)
            )
            image = axes[row, col].imshow(error, vmin=0, vmax=1, cmap="magma")
            r, c = np.unravel_index(np.argmax(error), error.shape)
            axes[row, col].plot(c, r, "+", color="cyan", ms=9)
            axes[row, col].set_title(LABELS[method] + " · pixel RMSE")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(image, ax=axes, shrink=0.7, label="Fraction RMSE · shared [0, 1] scale")
    fig.suptitle(
        "Reference disagreement · seed 0\nCyan marks each method's largest pixel error"
    )
    save(fig, target)


def measured_table(summary, synthetic=False):
    lines = [
        (
            "| Scene / case | Method | Runs | SAD ↓ (°) | Abundance RMSE ↓ | "
            "Reconstruction ↓ | Fit time (s) |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if (row["scene"] == "synthetic") != synthetic:
            continue

        def value(key, precision, row=row):
            mean, sd = row[key + "_mean"], row[key + "_std"]
            return f"{mean:.{precision}f}" + (f" ± {sd:.2g}" if sd is not None else "")

        title = row["case"] if synthetic else row["scene"].title()
        lines.append(
            f"| {title} | {LABELS[row['method']]} | {row['n_runs']} | "
            f"{value('mean_endmember_sad_deg', 2)} | {value('abundance_rmse', 3)} | "
            f"{value('relative_reconstruction_error', 3)} | {value('fit_seconds', 2)} |"
        )
    return "\n".join(lines)


def build_report(results, data_root, output):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    output = Path(output)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    record, runs = load_runs(results)
    summary = summarize(results)
    selections = representatives(results, record, data_root)
    for (name, case), selection in selections.items():
        if name != "synthetic" or case == "pure-snr35":
            abundance_figure(selection, assets / f"{name}-abundances.png")
    first = next(
        (v for (name, _), v in selections.items() if name == "samson"),
        next(iter(selections.values())),
    )
    overview_figure(first, assets / "overview.png")
    benchmark_figure(summary, assets / "benchmark.png")
    spectra_figure(selections, assets / "endmembers.png")
    synthetic_figure(summary, assets / "synthetic-benchmark.png")
    error_figure(selections, assets / "error-maps.png")
    iterative = [r for r in runs if r["method"] != "spa-fcls"]
    converged = sum(r["diagnostics"]["converged"] for r in iterative)
    findings = []
    for name in record["config"]["real_scenes"]:
        rows = [r for r in summary if r["scene"] == name]
        best_r = min(rows, key=lambda r: r["relative_reconstruction_error_mean"])
        best_a = min(rows, key=lambda r: r["abundance_rmse_mean"])
        findings.append(
            f"- **{name.title()}:** lowest reconstruction error: {LABELS[best_r['method']]} "
            f"({best_r['relative_reconstruction_error_mean']:.3f}); lowest abundance RMSE: "
            f"{LABELS[best_a['method']]} ({best_a['abundance_rmse_mean']:.3f})."
        )
    synthetic_findings = []
    for row in summary:
        if (
            row["scene"] == "synthetic"
            and row["method"] == "simplex-nmf"
            and row["case"] in ["pure-snr35", "pure-snr20", "mixed-clean"]
        ):
            synthetic_findings.append(
                f"- **{row['case']}, Simplex NMF:** SAD "
                f"{row['mean_endmember_sad_deg_mean']:.3f} degrees, abundance RMSE "
                f"{row['abundance_rmse_mean']:.4f}, reconstruction error "
                f"{row['relative_reconstruction_error_mean']:.6f}."
            )
    text = f"""# Experiment report

Generated from **{len(runs)} completed fits**. Configuration SHA-256:
`{record["config_sha256"]}`. Figures always use seed 0; every configured run is retained.

## Real-scene comparison

{measured_table(summary)}

NMF values are mean ± sample standard deviation over initialization seeds. SPA runs
once per real scene. Standard deviations are not confidence intervals and do not
measure transfer to other scenes. Full precision is retained in results/summary.csv.

![Measured comparison](assets/benchmark.png)

## What the measurements show

{chr(10).join(findings)}

Reconstruction and material recovery answer different questions. Low residuals do
not establish correct endmembers or fractions. Simplex constraints remove arbitrary
NMF scaling freedom, but blind unmixing can remain non-identifiable. Illumination
changes, uncertain references and nonconvex local minima also matter.

![Reference disagreement](assets/error-maps.png)

## Controlled synthetic experiment

Analytic spectra are mixed with exact fractions; they are not laboratory signatures.
Pure scenes contain pure patches; mixed scenes constrain every fraction to [0.1, 0.8].
Gaussian noise is scaled to the requested SNR before zero clipping. Realized SNR and
clipping counts are saved. Synthetic seeds vary texture, noise and initialization.

{chr(10).join(synthetic_findings)}

The high-SNR pure case checks recovery under favorable assumptions. At 20 dB, noisy
extreme-pixel selection deteriorates. In the noise-free mixed case, near-perfect
reconstruction coexists with fraction errors, illustrating non-identifiability.

{measured_table(summary, True)}

![Controlled mixtures](assets/synthetic-benchmark.png)

## Optimization and timing

{converged}/{len(iterative)} iterative fits met the objective-change stopping rule.
Others reached the fixed {record["config"]["max_iter"]}-step budget. Neither condition
proves a global optimum. Each run records objective traces and FCLS diagnostics.

Times include SPA, initialization and abundance fitting with {record["config"]["threads"]}
BLAS thread(s). They exclude downloads, source loading, reference matching, metrics
and figures. Hardware: {record["environment"]["cpu"]}; Python {record["environment"]["python"]}.
These small per-scene timings are not production throughput measurements.

## Reference and preprocessing caveats

- Samson has 95 × 95 pixels, 156 bands and three references. Source labels are
  1-rock, 2-Tree, 3-water; Rock/soil is used as the first display label.
- Jasper Ridge has 100 × 100 pixels, 198 retained bands and four references:
  1-tree, 2-water, 3-dirt, 4-road. Its retained bands are selected in the source file.
- MATLAB column-major pixel order is preserved. Jasper raw counts are divided by
  maxValue=5000, then each scene uses one global maximum scale. Fitting does not
  independently standardize bands. False color independently stretches display channels only.
- Real reference maps are benchmark estimates, not exact field measurements.
  Samson reference component amplitudes are separately normalized. Spectral angles
  compare shape; reference spectra are never a calibrated reconstruction oracle.
- Spectrum figures peak-normalize for display only. Wavelength calibration is not
  supplied, so real-scene plots use retained band indices.
- K is supplied and every scene pixel is available for unsupervised fitting. This is
  per-scene decomposition, not held-out classification or sensor-transfer testing.
- Hungarian spectral matching happens after fitting, for permutation alignment only.
  No reference-driven hyperparameter search or best-seed display is performed.
- Equal-weight spatial smoothing may erase small structures or cross boundaries.

See [methodology](https://github.com/amersaidi2002-byte/hyperspectral-unmixing/blob/main/docs/methodology.md)
and [source references](https://github.com/amersaidi2002-byte/hyperspectral-unmixing/blob/main/docs/references.md).
"""
    for name in ["benchmark", "error-maps", "synthetic-benchmark"]:
        if not (assets / f"{name}.png").exists():
            text = "\n".join(
                line for line in text.splitlines() if f"assets/{name}.png" not in line
            )
    (output / "experiment.md").write_text(text)
    (output / "measured-table.md").write_text(measured_table(summary) + "\n")
    print(f"Built report and figures in {output}")
