# Methodology

## Problem and conventions

Each hyperspectral pixel contains a spectrum. Linear unmixing estimates K component
spectra and their fractions: `X ≈ A @ E`. X is pixels × bands, E components × bands,
and A pixels × components. All estimates use float64, nonnegative E, and simplex
rows of A: nonnegative fractions summing to one. K is supplied (3 or 4).

The real loaders preserve MATLAB column-major pixel order. Jasper integer counts
are divided by the supplied maxValue=5000; each cube then uses one scene-global
maximum scale. Fitting never independently centers or standardizes spectral bands.
Preview-channel stretches are for display only.

## SPA + FCLS

SPA selects candidate pure pixels by successive orthogonal projection. Spectra are
L1-normalized only during selection, preventing brightness alone from dominating.
Selected pixels are returned in their original globally scaled units. Zero-energy
pixels cannot be selected. No spatial filtering or preconditioning is applied.
Pure-pixel availability is an assumption; noisy extreme pixels can be selected.

FCLS minimizes `0.5 * ||X - A @ E||²` with E fixed and simplex A. Objective-restarted
FISTA uses the largest eigenvalue of `E @ E.T` as its Lipschitz constant. Each step
uses a sorting-based Euclidean simplex projection. Clipping then normalizing an
unconstrained least-squares result is not the same optimization problem.

Every 20 steps, FCLS checks the maximum row norm of
`A - project_simplex(A - gradient/L)`. Defaults are tolerance 1e-8 and 3,000 steps.
Tests compare this implementation with independent SLSQP solutions and exact mixtures.

## Simplex-constrained NMF

The objective is `||X - A @ E||² / (2*N*B)`, with A on the simplex and E nonnegative.
SPA initializes E; seeded Gaussian perturbations have standard deviation
`0.02 * std(X)`, clipped above 1e-10. FCLS initializes A.

Each outer step performs three nonnegative projected-gradient E updates and 15
simplex-projected A updates. Gram matrices give the block Lipschitz constants.
Losses are recorded every five steps. The 300-step budget is fixed; early stopping
requires relative loss change below 1e-7 at four consecutive checks.

The simplex removes arbitrary factor rescaling but does not identify a unique
physical decomposition. NMF is nonconvex; neither stopping nor low residuals proves
correct endmembers, correct fractions or a global optimum.

## Spatial NMF

The same objective adds `lambda/(2*N) * sum_edges ||a_i-a_j||²`, lambda=0.002.
Edges connect horizontal/vertical neighbors once. The A gradient adds `lambda*L@A`
for the grid Laplacian L; 8 is a safe upper spectral bound. Image borders do not
wrap around. The reconstruction term is divided by B, keeping the lambda definition
consistent across band counts.

All spatial edges have equal weights. This intentionally simple baseline is not
total variation or edge-aware regularization: it can blur boundaries and small features.

## Evaluation

Only the observed cube, K and settings are passed to fitting. Afterwards, Hungarian
assignment aligns component permutations by the spectral-angle cost matrix. The same
permutation is applied to E and A. No reference-driven amplitude correction is fitted.

| Metric | Definition |
|---|---|
| Endmember SAD | Mean angle between matched reference/estimated spectra, in degrees |
| Abundance RMSE | Root mean squared fraction difference over all pixels and components |
| Reconstruction error | `norm(X-A@E) / norm(X)` |
| Reconstruction SAM | Mean pixel angle between observed/reconstructed nonzero spectra |
| Constraint errors | Minimum fraction and largest absolute sum-to-one error |

Samson reference spectral amplitudes are independently normalized; SAD compares
shape and does not require common amplitude units. These references are never a
calibrated reconstruction oracle. Reference abundance maps in both real scenes
are benchmark estimates, not exact field measurements. Synthetic references are exact.

## Protocol

Every real-scene pixel is available for unsupervised fitting. This is per-scene
decomposition, not held-out classification. Deterministic SPA runs once per real
scene; both NMF variants run at seeds 0–4. Tables retain all runs and use sample SD;
figures and the inspector always use seed 0. No best-restart selection is used.

Synthetic experiments cross pure/mixed regimes, noise-free/35 dB/20 dB conditions,
and five seeds. Pure patches provide pure pixels. Mixed fractions are in [0.1, 0.8].
Seeds jointly vary texture, noise and initialization. Gaussian noise is scaled to
the desired Frobenius-norm SNR before zero clipping; realized SNR and clipping are saved.

The fixed configuration is [benchmark.json](../configs/benchmark.json). Parameters
are not chosen by reference-label search. Timings include SPA, initialization and
fitting under one BLAS thread, excluding source loading, metrics and report rendering.

## Limits

More independent scenes, uncertain K, illumination-aware mixing and robust pure-pixel
selection are useful next studies. Equal-weight smoothing does not preserve every
boundary. These experiments do not establish sensor transfer or operational material
mapping, and synthetic noise tests do not substitute for independent acquisitions.
