# Reading the project

Start with the [executed notebook](../notebooks/01_synthetic_unmixing.ipynb). It builds
a cube with exact fractions and fits all three methods without a download or GPU.

Follow one pixel through the code:

1. `data.py` loads a `(height, width, bands)` cube, verifies source hashes, preserves
   MATLAB spatial order and applies a global scale. Preview stretches affect display only.
2. `algorithms.py` initializes K component spectra, fits FCLS fractions, and optionally
   refines both factors with constrained NMF. Every fraction update uses the simplex.
3. `metrics.py` aligns permutations after fitting, then measures spectral shape,
   fraction recovery and reconstruction separately.
4. `experiment.py` retains every configured run and seed-zero arrays for figures.
5. `explorer.py` serves saved outputs locally: observed spectra, component contributions,
   reconstructions and reference/estimated fractions.

Questions the experiments answer:

- Does improving reconstruction also improve fractions?
- What changes when pure pixels are absent?
- How does noise affect extreme-pixel selection?
- When does spatial smoothing help, and what structures can it erase?
- Which conclusions use exact synthetic references, and which use uncertain real references?

The [report](experiment.md) contains measured answers for the fixed configuration.
The [methodology](methodology.md) states the assumptions needed to interpret them.
