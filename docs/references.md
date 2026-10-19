# Sources and provenance

## Real data

Samson and Jasper Ridge MATLAB files come from the data repository associated with
Gaetano Settembre, Flavia Esposito and Nicoletta Del Buono, *Advancing blind
hyperspectral unmixing in remote sensing: comparing deep-inspired subspace learning
methods*, Advanced Modeling and Simulation in Engineering Sciences 12, 30 (2025).
[Paper](https://doi.org/10.1186/s40323-025-00313-6) ·
[Data repository](https://github.com/gaetanosettembre/data_unmixing).

Revision: `29774c39f00d385b1d7898a8b1f26b3710aa036b`. Exact URLs, sizes and SHA-256
hashes are in [sources.json](../configs/sources.json), also bundled with the package.
They identify this research snapshot rather than independently certifying original
sensor archives. Raw MATLAB files are downloaded on demand and not committed.

Reference-labeling context: Feiyun Zhu et al., *Hyperspectral Unmixing: Ground Truth
Labeling, Datasets, Benchmark Performances and Survey* (2017).
[Paper](https://arxiv.org/abs/1708.05125) ·
[Dataset page](https://lesun.weebly.com/hyperspectral-data-set.html).

Names/order are read from source `cood` entries: Samson 1-rock, 2-Tree, 3-water;
Jasper Ridge 1-tree, 2-water, 3-dirt, 4-road. Rock/soil is the first Samson display label.
Reference fractions are estimates rather than field measurements. Synthetic signals
are analytic functions from this code, not real laboratory spectral signatures.

## Algorithmic background

- Gillis and Vavasis, *Fast and Robust Recursive Algorithms for Separable Nonnegative
  Matrix Factorization*, IEEE TPAMI 36(4), 698–714 (2014).
  [Manuscript](https://arxiv.org/abs/1208.1237). SPA and the pure-pixel assumption.
- Lee and Seung, *Algorithms for Non-negative Matrix Factorization*, NeurIPS (2000).
  [Paper](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html).
  NMF background; the present solver uses projected gradients, not multiplicative updates.
- Beck and Teboulle, *A Fast Iterative Shrinkage-Thresholding Algorithm for Linear
  Inverse Problems*, SIAM Journal on Imaging Sciences 2(1), 183–202 (2009).
  [Paper](https://doi.org/10.1137/080716542). Acceleration background for FCLS.
- Duchi et al., *Efficient Projections onto the l1-Ball for Learning in High Dimensions*,
  ICML (2008). [Author PDF](https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf).
  Sorting/thresholding background for simplex projection.

These sources explain the building blocks. The compact alternating solver and
spatial variant are transparent benchmark implementations, not claims of exact
reproduction of a published state-of-the-art system. No external algorithm source
files were copied into the implementation. See [NOTICE](../NOTICE) for data terms.
